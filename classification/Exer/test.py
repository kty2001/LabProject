import argparse

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from src.model import EfficientNetB7Classifier
from src.model import EfficientNetB5Classifier
from src.model import EfficientNetB3Classifier
from src.model import EfficientNetB1Classifier
from src.model import ResNet152Classifier
from src.model import ResNet50Classifier
from src.model import ResNet18Classifier
from src.dataset import ExerciseDataset


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
parser.add_argument("--idx", default="0", help="테스트용 이미지 설정")
args = parser.parse_args()


def predict(test_data: Dataset, model: nn.Module, device: str, idx: int) -> None:
    model.eval()
    image = test_data[idx][0].to(device)
    image = image.unsqueeze(0)
    target = test_data[idx][1].to(device)
    with torch.no_grad():
        pred = model(image)
        predicted = pred[0].argmax(0)
        actual = target
        print(f'Predicted: "{test_data.classes[predicted]}", Actual: "{test_data.classes[actual]}"')
    return predicted, actual


def test(device, idx):

    if device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("<<< testing by", device, ">>>")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_data = ExerciseDataset("./images/test", transform=transform)
    num_classes = len(test_data.classes_dic)

    model = EfficientNetB7Classifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('exercise-net.pth'))

    img_pred, img_actual = predict(test_data, model, device, int(idx))

    return img_pred, img_actual


if __name__ == "__main__":
    test(device=args.device, idx=args.idx)
