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

train_dir = "../../data/images/train"
val_dir = "../../data/images/val"

# Hyperparameters
batch_size = 16
num_classes = len(os.listdir(train_dir))
num_epochs = 30
learning_rate = 0.001

# Data preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the size expected by ResNet50
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the last fully connected layer to match the number of classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer for better gradient updates

# Lists to store loss history for plotting
train_loss_history = []
val_loss_history = []

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0  # Initialize training loss for the epoch
    for images, labels in train_loader:  # Iterate through training batches
        images, labels = images.to(device), labels.to(device)  # Move images and labels to the device
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update model parameters
        train_loss += loss.item() * images.size(0)  # Accumulate loss, scaled by batch size

    train_loss /= len(train_loader.dataset)  # Average training loss for the epoch
    train_loss_history.append(train_loss)  # Save training loss for plotting

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_predictions = []  # Store predictions for accuracy calculation
    val_labels = []  # Store true labels for accuracy calculation
    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Compute validation loss
            val_loss += loss.item() * images.size(0)  # Accumulate loss
            _, preds = torch.max(outputs, 1)  # Get the predicted class indices
            val_predictions.extend(preds.cpu().numpy())  # Append predictions
            val_labels.extend(labels.cpu().numpy())  # Append true labels

    val_loss /= len(val_loader.dataset)  # Average validation loss
    val_loss_history.append(val_loss)  # Save validation loss for plotting

    val_acc = accuracy_score(val_labels, val_predictions)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "resnet50_pokemon.pth")

plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
