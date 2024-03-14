import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset import Cifar10Dataset
from src.model import LeNet
from src.utils import CLASSES, split_dataset


def predict(test_data: Dataset, model: nn.Module) -> None:
    """학습한 뉴럴 네트워크로 CIFAR-10 데이터셋 분류
    
    param test_data: 추론에 사용되는 데이터셋
    type test)data: Dataset
    param model: 추론에 사용되는 모델
    type model: nn.Module
    """
    model.eval()
    # image, target 설정
    image = test_data[0][0].unsqueeze(0)
    target = test_data[0][1]
    # 기울기 계산 생략
    with torch.no_grad():
        # 예측값과 실제값 비교
        pred = model(image)
        predicted = CLASSES[pred[0].argmax(0)]
        actual = CLASSES[target]
        print(f'Predicted: "{predicted}", Acutal: "{actual}"')

def test():
    # 검증 디렉토리 설정
    image_dir = 'data/train'
    test_csv_path = 'data/test_answer.csv'

    # 결과 레이블 개수 설정
    num_classes = 10

    # 텐서로 변경후 정규화 하도록 transform 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # test_data 초기화
    test_data = Cifar10Dataset(
        image_dir,
        test_csv_path,
        transform=transform
    )

    # 모델 및 가중치 불러오기
    model = LeNet(num_classes=num_classes)
    model.load_state_dict(torch.load('cifar-net-lenet.pth'))

    # 예측
    predict(test_data, model)

if __name__ =='__main__':
    test()