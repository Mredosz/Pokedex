import os
import shutil
from sklearn.model_selection import train_test_split

# Ścieżka do folderu z obrazami
data_dir = "../data/images"
train_dir = "../data/resnet/train"
val_dir = "../data/resnet/val"

# Tworzenie folderów train i val
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Podział danych
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if
                  img.endswith(('.jpg', '.png', '.jpeg'))]
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

        # Przenoszenie plików treningowych
        train_class_dir = os.path.join(train_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        for img in train_images:
            shutil.copy(img, train_class_dir)

        # Przenoszenie plików walidacyjnych
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(val_class_dir, exist_ok=True)
        for img in val_images:
            shutil.copy(img, val_class_dir)

print("Dane zostały podzielone na treningowe i walidacyjne.")
