import timm
import torch
from torchvision import transforms
from PIL import Image
from torchvision import datasets
import torch.nn as nn


def load_model_and_predict(image_path, model_path, num_classes):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda')

    model = timm.create_model('inception_v3', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        predicted_class = outputs.argmax(dim=-1).item()

    class_names = train_dataset.classes
    pokemon_name = class_names[predicted_class]

    print(f"Predicted Pokémon: {pokemon_name}")
    return pokemon_name


train_dataset = datasets.ImageFolder(root='../../data/images/train', transform=transforms.ToTensor())
image_path = "C:/Users/mredo/Downloads/Charizard%2C_the_Flame_Pokemon.png"
model_path = "inception_v3_pokemon_model.pth"
num_classes = len(train_dataset.classes)

pokemon_name = load_model_and_predict(image_path, model_path, num_classes)
