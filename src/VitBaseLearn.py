from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from PIL import Image

dataset = load_dataset("imagefolder", data_dir="../data/resnet")
device = torch.device('cuda')

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
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.001,
    per_device_train_batch_size=16,
    num_train_epochs=3,
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
