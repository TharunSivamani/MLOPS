import json
import os
import urllib
from io import BytesIO
from typing import Any

import requests
from cog import BasePredictor, Input, Path
from PIL import Image
from timm.models import create_model
from torch.nn.functional import softmax
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.model = create_model("efficientnet_b3a", pretrained=True)

    # Define the arguments and types the model takes as input
    def predict(self,image: Path = Input(description="Image to classify")) -> Any:
        """Run a single prediction on the model"""

        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)

        if str(image).startswith("https") or str(image).startswith("http"):
            url, filename = (image, "temp.jpg")
            urllib.request.urlretrieve(url, filename)
            img = Image.open(filename).convert('RGB')
            tensor = transform(img).unsqueeze(0) # transform and add batch dimension
        else:
            img = Image.open(image).convert('RGB')
            tensor = transform(img).unsqueeze(0) # transform and add batch dimension

        import torch
        with torch.no_grad():
            out = self.model(tensor)

        # Get imagenet class mappings
        url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
        urllib.request.urlretrieve(url, filename) 
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        probabilities = torch.nn.functional.softmax(out[0], dim=0)

        top_idx = torch.argmax(probabilities).item()
        top_prob = torch.max(probabilities)

        category = categories[top_idx]

        out = {"predicted": category, "confidence": top_prob}

        print(json.dumps(out))