from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import requests

# Load image from URL (sample dog image)
image_url = "https://images.unsplash.com/photo-1558788353-f76d92427f16"
image = Image.open(requests.get(image_url, stream=True).raw)

# Define text captions
captions = [
    "A cute dog in the grass",
    "A car driving down a street",
    "A person eating food"
]

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process inputs
inputs = processor(text=captions, images=image, return_tensors="pt", padding=True)

# Get predictions
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# Print results
print("\n=== Similarity Scores ===")
for i, caption in enumerate(captions):
    print(f"{caption} -> Score: {probs[0][i].item():.4f}")
