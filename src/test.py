from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from PIL import Image

# 1. Wczytanie zbioru danych obrazów Pokémonów
dataset = load_dataset("imagefolder", data_dir="../data/resnet")
device = torch.device('cuda')

# 2. Przygotowanie modelu i feature extractora
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(dataset['train'].features['label'].names),
    id2label={i: label for i, label in enumerate(dataset['train'].features['label'].names)},
    label2id={label: i for i, label in enumerate(dataset['train'].features['label'].names)},
    ignore_mismatched_sizes=True
)

model.to(device)

# 3. Przetwarzanie obrazów
def preprocess_images(example):
    example['image'] = example['image'].convert("RGB")

    # Konwersja do formatu PIL, jeśli to konieczne
    if not isinstance(example['image'], Image.Image):
        example['image'] = Image.open(example['image']).convert("RGB")

    # Dodanie obsługi wyjątków
    try:
        example['pixel_values'] = feature_extractor(images=example['image'], return_tensors="pt")['pixel_values'][0]
    except Exception as e:
        print(f"Error processing image: {e}")
        print(f"Image type: {type(example['image'])}")
        print(f"Image size: {example['image'].size}")
        raise e
    return example


dataset = dataset.map(preprocess_images, batched=False)

# 4. Ustawienia treningu
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
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
    tokenizer=feature_extractor,
    data_collator=lambda data: {
        'pixel_values': torch.stack([f["pixel_values"] for f in data]),
        'labels': torch.tensor([f["label"] for f in data])
    }
)

# 5. Trening
trainer.train()

# 6. Funkcja do predykcji klasy
def predict_pokemon(image_path, model, feature_extractor):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()
    pokemon_name = model.config.id2label[predicted_class]
    print(f"Predicted Pokémon: {pokemon_name}")

# Przykład użycia
predict_pokemon("test_image.jpg", model, feature_extractor)
