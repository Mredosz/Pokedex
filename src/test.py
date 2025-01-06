from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from PIL import Image
import pandas as pd

# 1. Wczytanie zbioru danych obrazów Pokémonów
# Zakładamy, że obrazy są w folderze `pokemon_images` z podfolderami dla każdej klasy.
dataset = load_dataset("imagefolder", data_dir="../data/images")

# 2. Wczytanie CSV z informacjami o Pokémonach
pokemon_info = pd.read_csv("../data/pokemon.csv")

# 3. Przygotowanie modelu i feature extractora
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(dataset['train'].features['label'].names),
    id2label={i: label for i, label in enumerate(dataset['train'].features['label'].names)},
    label2id={label: i for i, label in enumerate(dataset['train'].features['label'].names)},
    ignore_mismatched_sizes=True
)

# 4. Przetwarzanie obrazów
def preprocess_images(example):
    # Konwersja do formatu RGB, jeśli obraz jest w formacie plikowym
    if isinstance(example['image'], str):
        example['image'] = Image.open(example['image']).convert("RGB")

    # Przetwarzanie obrazu za pomocą feature_extractor
    example['pixel_values'] = feature_extractor(images=example['image'], return_tensors="pt")['pixel_values'][0]
    return example

dataset = dataset.map(preprocess_images, batched=False)

# 5. Ustawienia treningu
training_args = TrainingArguments(
    # output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    # logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
    data_collator=lambda data: {'pixel_values': torch.stack([f["pixel_values"] for f in data]),
                                'labels': torch.tensor([f["label"] for f in data])}
)

# 6. Trening
trainer.train()

# Funkcja do predykcji klasy i wypisania informacji z CSV
def predict_pokemon(image_path, model, feature_extractor, csv_data):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()

    # Pobierz nazwę Pokémona
    pokemon_name = model.config.id2label[predicted_class]

    # Znajdź informacje w CSV
    info = csv_data[csv_data['name'] == pokemon_name]
    print(f"Predicted Pokémon: {pokemon_name}")
    print("Information:")
    print(info)


# Przykład użycia
predict_pokemon("test_image.jpg", model, feature_extractor, pokemon_info)
