import json
from pathlib import Path


def parse_metrics_file(file_path):
    metrics = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("Accuracy:"):
                    metrics["Accuracy"] = float(line.split(":")[1].strip())
                elif line.startswith("Precision:"):
                    metrics["Precision"] = float(line.split(":")[1].strip())
                elif line.startswith("Recall:"):
                    metrics["Recall"] = float(line.split(":")[1].strip())
                elif line.startswith("F1-score:"):
                    metrics["F1-score"] = float(line.split(":")[1].strip())
    except Exception as e:
        metrics["Error"] = f"Failed to parse {file_path}: {str(e)}"
    return metrics


def aggregate_metrics(base_dir):
    aggregated_metrics = {}
    models = ["google_vit_base", "inception_v3", "resnet_50"]
    for model in models:
        metrics_file = base_dir / model / "metrics.txt"
        if metrics_file.exists():
            aggregated_metrics[model] = parse_metrics_file(metrics_file)
        else:
            aggregated_metrics[model] = {"Error": "Metrics file not found"}
    return aggregated_metrics


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    results = aggregate_metrics(base_dir)

    print(json.dumps(results, indent=4))
