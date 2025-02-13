### **Key Contributions & Achievements**

1. **Comprehensive Evaluation of CLIP’s Zero-Shot Classification**
    - Tested CLIP across diverse datasets (CIFAR-100, Food101, Oxford-IIIT Pets) to benchmark zero-shot performance, achieving 63.4%, 75.2%, and 80.08% accuracy, respectively.
    - Identified key challenges: **feature overlap** (e.g., misclassifying "Spaghetti Bolognese" as "Lasagna") and **lack of fine-grained representation** (e.g., confusion between "Persian Cat" and "Maine Coon").
    - Conducted error analysis to reveal model limitations in low-resolution images, ambiguous poses, and contextual gaps.
2. **Innovative Methodologies for Enhancing CLIP**
    - **Hierarchical Superclass Integration**: Adapted CHiLS framework to embed superclass context into prompts (e.g., “a photo of a bee, a type of insect”), though improvements were marginal.
    - **Training Strategy Optimization**: Experimented with Triplet Loss and queue-based contrastive learning, validating CLIP’s reliance on original contrastive loss for optimal performance.
3. **CLIPText: Zero-Shot Text Classification Framework**
    - Redefined text classification as a text-image matching problem, leveraging CLIP’s multimodal embeddings.
    - Developed **PROMPT-CLIPText**, which integrates task-specific hard prompts (e.g., “Topic: [Label]”) to boost accuracy from **18.0% to 43.0%** on AGNews, outperforming traditional models (XGBoost) in zero-shot settings.
    - Demonstrated resource efficiency: **No training data or time required**, enabling rapid deployment.
4. **Generalization to Novel Domains**
    - Validated CLIP’s robustness by testing on anime-inspired “catgirl” images (sourced from *Nekopara*), achieving >85% confidence in classification despite stylistic novelty.
    - Highlighted CLIP’s ability to infer semantic essence beyond training data, even with expanded non-cat categories (anthropomorphic characters, objects).

---

### **Technical Skills Demonstrated**

- **Model Architecture**: CLIP (ResNet/ViT, GPT-2), contrastive learning, Triplet Loss.
- **Applications**: Zero-shot image/text classification, cross-modal alignment, fine-grained recognition.
- **Tools**: PyTorch, NLP/vision datasets (CIFAR, Food101), prompt engineering.

--- 
