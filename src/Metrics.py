import json
from pathlib import Path


def parse_metrics_file(file_path):
    metrics = {}
    try:
        with open(file_path, 'r') as file:  # Open the metrics file in read mode
            for line in file:  # Iterate through each line in the file
                if line.startswith("Accuracy:"):
                    metrics["Accuracy"] = float(line.split(":")[1].strip())  # Extract and parse Accuracy
                elif line.startswith("Precision:"):
                    metrics["Precision"] = float(line.split(":")[1].strip())  # Extract and parse Precision
                elif line.startswith("Recall:"):
                    metrics["Recall"] = float(line.split(":")[1].strip())  # Extract and parse Recall
                elif line.startswith("F1-score:"):
                    metrics["F1-score"] = float(line.split(":")[1].strip())  # Extract and parse F1-score
    except Exception as e:
        metrics["Error"] = f"Failed to parse {file_path}: {str(e)}"
    return metrics


def aggregate_metrics(base_dir):
    aggregated_metrics = {}
    models = ["google_vit_base", "inception_v3", "resnet_50"]  # List of models to aggregate metrics for
    for model in models:
        metrics_file = base_dir / model / "metrics.txt"  # Construct the path to the metrics file
        if metrics_file.exists():  # Check if the metrics file exists
            aggregated_metrics[model] = parse_metrics_file(metrics_file)  # Parse and store the metrics
        else:
            aggregated_metrics[model] = {"Error": "Metrics file not found"}  # Handle missing metrics file
    return aggregated_metrics


if __name__ == "__main__":
    # Determine the base directory dynamically based on the script's location
    base_dir = Path(__file__).resolve().parent

    # Aggregate metrics for all models
    results = aggregate_metrics(base_dir)

    # Print the aggregated metrics as a formatted JSON string
    print(json.dumps(results, indent=4))
