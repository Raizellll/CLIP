import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet50 requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

class ImageEncoder(nn.Module):
    def __init__(self, mode='finetune'):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.resnet = resnet50(weights=weights)
        
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1))

    def forward(self, x):
        x = self.resnet(x)
        x = self.adaptive_pool(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        return x

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        self.model = GPT2Model.from_pretrained('openai-community/gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, -1, :]

class CLIP(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        # Projection layers
        self.image_projection = nn.Linear(2048, embedding_dim)  # ResNet50 outputs 2048-dim
        self.text_projection = nn.Linear(768, embedding_dim)    # GPT2 outputs 768-dim
        
        # Additional linear layers
        self.image_layer1 = nn.Linear(embedding_dim, embedding_dim)
        self.text_layer1 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, image, text):
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(text)
        
        image_embedding = self.image_projection(image_embedding)
        image_embedding = self.image_layer1(image_embedding)
        
        text_embedding = self.text_projection(text_embedding)
        text_embedding = self.text_layer1(text_embedding)
        
        # Normalize embeddings
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        
        return image_embedding, text_embedding

class EmbeddingQueue:
    def __init__(self, max_size=30):
        self.max_size = max_size
        self.img_queue = []
        self.text_queue = []
        
    def add_queue(self, img_emb, text_emb):
        self.img_queue.append(img_emb)
        self.text_queue.append(text_emb)
        
        if len(self.img_queue) > self.max_size:
            self.img_queue.pop(0)
            self.text_queue.pop(0)
            
    def return_values(self):
        return self.img_queue, self.text_queue

def train_one_epoch(model, dataloader, optimizer, embedding_queue=None, use_triplet_loss=False):
    model.train()
    total_loss = 0
    margin = 0.2  # For triplet loss
    
    for images, labels in dataloader:
        images = images.to(device)
        # Generate text descriptions
        texts = [f"This is an image of a {train_dataset.classes[label]}" for label in labels]
        
        optimizer.zero_grad()
        img_emb, text_emb = model(images, texts)
        
        if use_triplet_loss:
            # Triplet Loss implementation
            img_text_similarity = torch.matmul(img_emb, text_emb.t()).diag()
            n = img_emb.shape[0]
            original_list = list(range(n))
            shifted_list = original_list[1:] + [original_list[0]]
            shuffled_image = img_emb[shifted_list]
            shuffled_text = text_emb[shifted_list]
            neg_sim_img = torch.matmul(img_emb, shuffled_text.t()).diag()
            neg_sim_text = torch.matmul(text_emb, shuffled_image.t()).diag()
            img_loss = torch.clamp(margin + neg_sim_img - img_text_similarity, min=0)
            text_loss = torch.clamp(margin + neg_sim_text - img_text_similarity, min=0)
            loss = (img_loss.mean() + text_loss.mean())/2
        else:
            # Multi-Class N-pair Loss with queue
            if embedding_queue is not None:
                embedding_queue.add_queue(img_emb.clone().detach().cpu(), text_emb.clone().detach().cpu())
                saved_img_embeddings, saved_text_embeddings = embedding_queue.return_values()
                
                if len(saved_img_embeddings) > 0:
                    new_img = torch.cat(saved_img_embeddings, dim=0).to(device)
                    new_text = torch.cat(saved_text_embeddings, dim=0).to(device)
                    img_mat = torch.cat([img_emb, new_img], dim=0)
                    text_mat = torch.cat([text_emb, new_text], dim=0)
                else:
                    img_mat = img_emb
                    text_mat = text_emb
            else:
                img_mat = img_emb
                text_mat = text_emb
            
            labels = torch.arange(img_emb.shape[0]).to(device)
            logits_img_text = torch.matmul(img_emb, text_mat.t()) * 2
            logits_text_img = torch.matmul(text_emb, img_mat.t()) * 2
            
            img_text_loss = F.cross_entropy(logits_img_text, labels)
            text_img_loss = F.cross_entropy(logits_text_img, labels)
            loss = (img_text_loss + text_img_loss)/2
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataset, batch_size=100):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get all class text descriptions
            class_texts = [f"This is an image of a {dataset.classes[i]}" 
                         for i in range(len(dataset.classes))]
            
            # Get embeddings
            img_emb, _ = model(images, class_texts)
            _, text_emb = model(images[0:1], class_texts)  # Only need text encoding once
            
            # Calculate similarity and predict
            similarity = torch.matmul(img_emb, text_emb.t())
            predictions = similarity.argmax(dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    return correct / total * 100

def main():
    # Hyperparameters
    batch_size = 64
    epochs = 5
    learning_rate = 1e-4
    embedding_dim = 512
    use_queue = True
    use_triplet_loss = False
    
    # Initialize model and move to device
    model = CLIP(embedding_dim=embedding_dim).to(device)
    
    # Initialize queue if using it
    embedding_queue = EmbeddingQueue() if use_queue else None
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    
    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, 
                                   embedding_queue, use_triplet_loss)
        scheduler.step()
        
        # Evaluate
        accuracy = evaluate(model, test_dataset)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    main() 