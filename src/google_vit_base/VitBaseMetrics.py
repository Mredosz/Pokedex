from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
from VitBaseTest import predict_image

def evaluate_model(dataset_dir, output_file):
    true_labels = []
    predicted_labels = []

    print("Starting model evaluation...")

    dataset_dir = Path(dataset_dir)
    total_classes = len([d for d in dataset_dir.iterdir() if d.is_dir()])
    processed_classes = 0

    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            true_class = class_dir.name.lower()
            print(f"Processing class: {true_class}...")

            for image_file in class_dir.iterdir():
                if image_file.suffix in [".jpg", ".png"]:
                    predicted_class = predict_image(image_file).lower()

                    true_labels.append(true_class)
                    predicted_labels.append(predicted_class)

            processed_classes += 1
            print(f"Progress: {processed_classes}/{total_classes} classes processed.")

    print("Calculating results...")
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    print("Saving results to file...")
    with open(output_file, "w") as file:
        file.write("General score:\n")
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall:    {recall:.2f}\n")
        file.write(f"F1-score:  {f1:.2f}\n")

    print(f"Results saved to file: {output_file}")
    print("Model evaluation completed.")

project_dir = Path(__file__).resolve().parents[2]

dataset_dir = project_dir / "data" / "dataset"
output_file = project_dir / "src" / "google_vit_base" / "metrics.txt"

evaluate_model(dataset_dir, output_file)