from pathlib import Path
from resnet_50.ResnetTest import predict_image as predict_resnet
from inception_v3.InceptionTest import predict_image as predict_inception


def predict_with_all_models(image_path):
    image_path = Path(image_path)  # Ensure the image path is a Path object for compatibility

    try:
        resnet_result = predict_resnet(image_path)  # Make a prediction using the ResNet model
    except Exception as e:
        resnet_result = f"Error: {str(e)}"

    try:
        inception_result = predict_inception(image_path)  # Make a prediction using the Inception model
    except Exception as e:
        inception_result = f"Error: {str(e)}"

    return {
        "ResNet": resnet_result,
        "Inception": inception_result,
    }


if __name__ == "__main__":
    import sys
    import json

    # Check if the script is run with an image path as an argument
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]  # Get the image path from the command-line arguments
    results = predict_with_all_models(image_path)
    print(json.dumps(results))
