import argparse

import torch
from torch import nn
from torch.utils.data import Dataset

from src.model import ResNet18Classifier
from src.dataset import ExerciseDataset


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()


def predict(test_data: Dataset, model: nn.Module, device: str) -> None:
    model.eval()
    image = test_data[0][0].to(device)
    image = image.unsqueeze(0)
    target = test_data[0][1].to(device)
    with torch.no_grad():
        pred = model(image)
        predicted = pred[0].argmax(0)
        actual = target
        print(f'Predicted: "{test_data.classes[predicted]}", Actual: "{test_data.classes[actual]}"')


def test(device):
    num_classes = 5

    if device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)

    test_data = ExerciseDataset("./images/test")
    model = ResNet18Classifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('exercise-resnet18.pth'))

    predict(test_data, model, device)


if __name__ == "__main__":
    test(device=args.device)
