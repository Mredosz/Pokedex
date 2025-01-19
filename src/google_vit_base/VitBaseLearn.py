import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from PIL import Image
import torch.nn as nn

train_dir = "../../data/images/train"
val_dir = "../../data/images/val"

# Hyperparameters
batch_size = 16
num_classes = len(os.listdir(train_dir))
num_epochs = 30
learning_rate = 0.001

# Define data transformation pipeline
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize images to match input requirements for pre-trained models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize values to match pre-trained model expectations
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset using the Hugging Face `datasets` library
dataset = load_dataset("imagefolder", data_dir="../../data/images")

# Load the ViT model and feature extractor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')  # Pre-trained feature extractor for ViT
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(dataset['train'].features['label'].names),  # Map labels dynamically based on the dataset
    id2label={i: label for i, label in enumerate(dataset['train'].features['label'].names)},  # Create mapping from ID to label
    label2id={label: i for i, label in enumerate(dataset['train'].features['label'].names)},  # Create mapping from label to ID
    ignore_mismatched_sizes=True  # Allow model to handle mismatched dimensions if needed
)

model.to(device)

# Preprocess the dataset to handle image transformations
def preprocess_images(example):
    example['image'] = example['image'].convert("RGB")

    # Handle cases where the image might not be a PIL Image
    if not isinstance(example['image'], Image.Image):  # Check if the image is not already a PIL image
        example['image'] = Image.open(example['image']).convert("RGB")  # Open and convert to RGB if necessary

    # Apply the feature extractor to convert images into pixel values
    try:
        example['pixel_values'] = feature_extractor(images=example['image'], return_tensors="pt")['pixel_values'][0]
    except Exception as e:
        # Log detailed error information for debugging
        print(f"Error processing image: {e}")
        print(f"Image type: {type(example['image'])}")  # Log type of image for troubleshooting
        print(f"Image size: {example['image'].size}")  # Log image size for further debugging
        raise e
    return example

dataset = dataset.map(preprocess_images, batched=False)

training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=lambda data: {
        # Stack pixel values into a batch tensor
        'pixel_values': torch.stack([torch.tensor(f["pixel_values"]) for f in data]),
        # Create a tensor of labels
        'labels': torch.tensor([f["label"] for f in data])
    }
)


trainer.train()
