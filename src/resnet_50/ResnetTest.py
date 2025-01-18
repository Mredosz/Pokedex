from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import os
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

def predict_image(image_path):
    project_dir = Path(__file__).resolve().parents[2]

    device = torch.device('cuda')
    train_dir = project_dir / "data" / "images" / "train"
    model_dir = project_dir / "src" / "resnet_50" / "resnet50_pokemon.pth"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(os.listdir(train_dir)))

    model.load_state_dict(torch.load(model_dir, map_location=device))

    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return train_dataset.classes[predicted.item()]