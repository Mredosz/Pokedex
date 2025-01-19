from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
import torch
from torch import optim
import os

train_dir = "../../data/images/train"
val_dir = "../../data/images/val"

# Hyperparameters
batch_size = 16
num_classes = len(os.listdir(train_dir))
num_epochs = 30
learning_rate = 0.001

# Data transformation pipeline for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize images to match InceptionV3's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pretrained models
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the pretrained InceptionV3 model
model = timm.create_model('inception_v3', pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the final fully connected layer to match the number of classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss is suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer for efficient training

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:  # Iterate over batches of training data
        images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model

        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss

        optimizer.zero_grad()  # Clear gradients from the previous step
        loss.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update model parameters

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation for evaluation
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the device
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class with the highest score
        total += labels.size(0)  # Accumulate the total number of samples
        correct += (predicted == labels).sum().item()  # Accumulate the number of correct predictions

print(f'Accuracy: {100 * correct / total:.2f}%')

torch.save(model.state_dict(), 'inception_v3_pokemon_model.pth')
print("Model saved to 'inception_v3_pokemon_model.pth'")

