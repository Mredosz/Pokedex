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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = project_dir / "data" / "images" / "train"
    model_dir = project_dir / "src" / "resnet_50" / "resnet50_pokemon.pth"

    # Define the transformation pipeline for preprocessing the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to the size expected by ResNet50
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
    ])

    # Load the training dataset to retrieve class labels
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)  # Automatically assigns class labels based on folder structure

    model = models.resnet50(weights=None)  # Initialize without pre-trained weights
    model.fc = nn.Linear(model.fc.in_features, len(os.listdir(train_dir)))  # Replace the final layer to match the number of classes

    model.load_state_dict(torch.load(model_dir, map_location=device, weights_only=True))  # Map model to the appropriate device

    model = model.to(device)
    model.eval()

    # Load and preprocess the input image
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add a batch dimension

    # Perform inference
    with torch.no_grad():  # Disable gradient computation for faster and memory-efficient inference
        outputs = model(image)  # Forward pass through the model
        _, predicted = torch.max(outputs, 1)  # Get the class index with the highest predicted score

    # Map the predicted index to the corresponding class name
    return train_dataset.classes[predicted.item()]