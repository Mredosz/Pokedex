from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

def load_model_and_predict(image_path, model_path, feature_extractor_path):
    feature_extractor = ViTImageProcessor.from_pretrained(feature_extractor_path)

    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()
    pokemon_name = model.config.id2label[predicted_class]

    print(f"Predicted Pokémon: {pokemon_name}")
    return pokemon_name


image_path = "../../data/images/Dugtrio/0c5f972fb2c64e7f8468ef44c98ff3e5.jpg"

model_path = "../results/checkpoint-1011"
feature_extractor_path = 'google/vit-base-patch16-224'

pokemon_name = load_model_and_predict(image_path, model_path, feature_extractor_path)
