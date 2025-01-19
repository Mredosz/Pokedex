from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

def predict_image(image_path):
    project_dir = Path(__file__).resolve().parents[2]

    model_path = project_dir / "src" / "google_vit_base" / "results" / "checkpoint-10110"
    feature_extractor_path = 'google/vit-base-patch16-224'
    feature_extractor = ViTImageProcessor.from_pretrained(feature_extractor_path)

    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()
    pokemon_name = model.config.id2label[predicted_class]

    return pokemon_name