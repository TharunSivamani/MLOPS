import argparse
import io
import json

import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from timm.models import create_model


def main(model: str, image: str):
    
    labels2class = json.load(open("imagenet1000_labels.json", "r"))

    model = create_model(model, pretrained=True)
    model.eval()

    if "http" in image:
        response = requests.get(image)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = transform(image)
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)
        prob = torch.nn.functional.softmax(pred, dim=1)
        idx = prob.argmax().item()

    print(json.dumps({"predicted": labels2class[str(idx)], "confidence": pred[0,idx].item()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="resnet152", help="model name needed for inferencing"
    )
    parser.add_argument(
        "--image",
        default="https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        help="image path is required for inferencing",
    )
    args = parser.parse_args()

    main(model=args.model, image=args.image)
