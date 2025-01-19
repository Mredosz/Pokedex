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

batch_size = 16
num_classes = len(os.listdir(train_dir))
num_epochs = 30
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("imagefolder", data_dir="../../data/images")

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(dataset['train'].features['label'].names),
    id2label={i: label for i, label in enumerate(dataset['train'].features['label'].names)},
    label2id={label: i for i, label in enumerate(dataset['train'].features['label'].names)},
    ignore_mismatched_sizes=True
)

model.to(device)

def preprocess_images(example):
    example['image'] = example['image'].convert("RGB")

    if not isinstance(example['image'], Image.Image):
        example['image'] = Image.open(example['image']).convert("RGB")

    try:
        example['pixel_values'] = feature_extractor(images=example['image'], return_tensors="pt")['pixel_values'][0]
    except Exception as e:
        print(f"Error processing image: {e}")
        print(f"Image type: {type(example['image'])}")
        print(f"Image size: {example['image'].size}")
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
        'pixel_values': torch.stack([torch.tensor(f["pixel_values"]) for f in data]),
        'labels': torch.tensor([f["label"] for f in data])
    }
)


trainer.train()
