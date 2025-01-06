from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from PIL import Image
import os
import torch

# Wczytaj dane
def load_pokemon_dataset(json_path):
    dataset = load_dataset("json", data_files=json_path)
    return dataset["train"]

# Przygotowanie modelu i procesora
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Funkcja do przetwarzania danych
def preprocess_data(examples):
    # Konstruowanie pełnej ścieżki do obrazu
    image_path = examples["image_path"]  # Ścieżka już zaczyna się od "data/images/"

    # Wczytanie obrazu za pomocą PIL
    try:
        image = Image.open(image_path).convert("RGB")  # Upewniamy się, że obraz jest w formacie RGB
    except FileNotFoundError:
        raise FileNotFoundError(f"Nie znaleziono pliku obrazu: {image_path}")

    # Przetwarzanie obrazu za pomocą processor
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    return {"pixel_values": pixel_values, "caption": examples["caption"]}

# Wczytaj i przetwórz dane
dataset = load_pokemon_dataset("../data/pokemon_formatted_training_data.json")
dataset = dataset.map(preprocess_data)

# Ustawienia treningu
training_args = TrainingArguments(
    output_dir="./blip_pokemon",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
)

# Trenowanie modelu
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()


# Wczytaj fine-tunowany model
model = BlipForConditionalGeneration.from_pretrained("./blip_pokemon")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

# Generowanie opisu
image = Image.open("images/pikachu.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")
caption_ids = model.generate(**inputs)
caption = processor.decode(caption_ids[0], skip_special_tokens=True)
print("Generated Caption:", caption)
