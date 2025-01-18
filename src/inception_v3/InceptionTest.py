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

    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transforms.ToTensor())
    num_classes = len(train_dataset.classes)

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

    return pokemon_name