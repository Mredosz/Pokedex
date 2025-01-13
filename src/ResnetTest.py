from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy
device = torch.device('cuda')
# Tworzenie nowej instancji modelu
train_dir = "../data/resnet/train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)


model = models.resnet50(weights=None)  # Nie ładuj pretrenowanych wag
model.fc = nn.Linear(model.fc.in_features, len(os.listdir(train_dir)))  # Dostosowanie liczby klas

# Wczytaj zapisane wagi modelu
model.load_state_dict(torch.load("resnet50_pokemon.pth", map_location=device))

# Ustaw model w tryb oceny
model = model.to(device)
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return train_dataset.classes[predicted.item()]

# Przykład użycia
image_path = "../data/images/Dugtrio/0c5f972fb2c64e7f8468ef44c98ff3e5.jpg"
predicted_class = predict_image(image_path)
print(f"Predicted Pokémon: {predicted_class}")
