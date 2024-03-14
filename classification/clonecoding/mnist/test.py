import argparse

import torch
from torch import nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from src.model import NeuralNetwork
from src.dataset import MnistDataset

# 파서 설정
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()

# 예측 함수
def predict(test_data: Dataset, model: nn.Module, device: str) -> None:
    """학습한 뉴럴 네트워크로 MNIST 데이터셋 분류
    
    param tet_data: 추론에 사용되는 데이터셋
    type test_data: Dataset
    param model: 추론에 사용되는 모델
    type model: nn.Module
    """

    model.eval()
    # 데이터 불러오기 및 gpu로 이동
    image = test_data[0][0]
    plt.imshow(image)
    plt.savefig("output_image.png")
    image = image.to(device)
    image = image.unsqueeze(0)
    target = test_data[0][1].to(device)
    # 예측
    with torch.no_grad():
        pred = model(image)
        predicted = pred[0].argmax(0)
        actual = target
        print(f'Predicted: "{predicted}", Acutal: "{actual}"')
        
# 테스트
def test(device):
    # num_classes 값 설정
    num_classes = 10

    # test용 데이터 초기화
    test_data = MnistDataset("./data/MNIST Dataset JPG format/MNIST - JPG - testing", transform=None)
    # print(test_data[0][0].shape)

    # 모델 설정 및 gpu로 이동 및 파라미터값 불러오기
    model = NeuralNetwork(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('mnist-net.pth'))

    # 예측
    predict(test_data, model, device)

if __name__ == "__main__":
    test(device=args.device)