from pathlib import Path
from resnet_50.ResnetTest import predict_image as predict_resnet
from inception_v3.InceptionTest import predict_image as predict_inception
from google_vit_base.VitBaseTest import predict_image as predict_vit

def predict_with_all_models(image_path):
    image_path = Path(image_path)

    try:
        resnet_result = predict_resnet(image_path)
    except Exception as e:
        resnet_result = f"Error: {str(e)}"

    try:
        inception_result = predict_inception(image_path)
    except Exception as e:
        inception_result = f"Error: {str(e)}"

    try:
        vit_result = predict_vit(image_path)
    except Exception as e:
        vit_result = f"Error: {str(e)}"

    return {
        "ResNet": resnet_result,
        "Inception": inception_result,
        "ViT": vit_result
    }

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]
    results = predict_with_all_models(image_path)
    print(json.dumps(results))
