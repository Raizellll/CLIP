import requests
from PIL import Image
from io import BytesIO
import os

# Create emotion_images directory if it doesn't exist
os.makedirs('emotion_images', exist_ok=True)

# Emoji image URLs for each emotion
emotion_image_urls = {
    'sadness': 'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f622.png',  # Crying face
    'joy': 'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f604.png',      # Smiling face with open mouth and smiling eyes
    'love': 'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/2764.png',      # Red heart
    'anger': 'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f621.png',    # Pouting face
    'fear': 'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f628.png',     # Fearful face
    'surprise': 'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f632.png'  # Astonished face
}

def download_image(url, filename):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize to a standard size
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img.save(f'emotion_images/{filename}.jpg', quality=95)
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Failed to download {filename}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")

# Download images
for emotion, url in emotion_image_urls.items():
    download_image(url, emotion)
    
print("\nDownload complete. Please check the emotion_images directory.") 