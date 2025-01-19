import timm
import torch
from torchvision import transforms
from PIL import Image
from torchvision import datasets
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

def predict_image(image_path):
    project_dir = Path(__file__).resolve().parents[2]

    train_dataset_path = project_dir / "data" / "images" / "train"
    model_path = project_dir / "src" / "inception_v3" / "inception_v3_pokemon_model.pth"

    # Load the training dataset to retrieve class names
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transforms.ToTensor())
    num_classes = len(train_dataset.classes)

    # Define the transformation pipeline for preprocessing the image
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize image to match InceptionV3 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pretrained model
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model('inception_v3', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()  # Move model to device and set to evaluation mode

    # Load and preprocess the input image
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

    # Perform inference with the model
    with torch.no_grad():  # Disable gradient computation for faster and memory-efficient inference
        outputs = model(image)  # Forward pass through the model
        predicted_class = outputs.argmax(dim=-1).item()  # Get the index of the class with the highest score

    # Map the predicted index to the corresponding class name
    class_names = train_dataset.classes  # Retrieve class names from the dataset
    pokemon_name = class_names[predicted_class]  # Map the predicted index to the class name

    return pokemon_name