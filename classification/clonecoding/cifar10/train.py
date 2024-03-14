import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import Cifar10Dataset
from src.model import LeNet
from src.utils import split_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()

def train_one_epoch(dataloader: DataLoader, device, model: nn.Module, loss_fn: nn.Module, optimizer) -> None:
    """CIFAR-10 데이터셋으로 뉴럴 네트워크를 훈련
    
    param dataloader: 파이토치 데이터로더
    type dataloader: DataLoader
    param device: 훈련에 사용되는 장치
    type device: _device
    param model: 훈련에 사용되는 모델
    type model: nn.Module
    param loss_fn: 훈련에 사용되는 오차 함수
    type loss_fn: nn.Module
    param
    """

    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        # GPU로 이동
        images = images.to(device)
        targets = targets.to(device)

        # 예측 및 오차 계산
        preds = model(images)
        loss = loss_fn(preds, targets)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 100번 째 batch마다 loss와 진행상황 출력
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def val_one_epoch(dataloader: DataLoader, device, model: nn.Module, loss_fn: nn.Module) -> None:
    """CIFAR-10 데이터셋으로 뉴럴 네트워크 성능 테스트
    
    param dataloader: 파이토치 데이터로더
    type dataloader: DataLoader
    param device: 훈련에 사용되는 장치
    type device: _device
    param model: 훈련에 사용되는 모델
    type model: nn.Module
    param loss_fn: 훈련에 사용되는 오차 함수
    type loss_fn: nn.Module
    param
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    # 기울기 계산 생략
    with torch.no_grad():
        for images, targets in dataloader:
            # GPU로 이동
            images = images.to(device)
            targets = targets.to(device)
            
            # 예측
            preds = model(images)

            # loss 및 정답 합산
            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).float().sum().item()
    
    # 배치당 평균 loss와 정답률 출력
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

def train(device) -> None:
    """학습/추론 파이토치 파이프라인
    
    param batch_size: 학습 및 추론 데이터셋의 배치 크기
    type batch_size: int
    param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    type epochs: int
    """
    
    # 디렉토리 설정
    image_dir = 'data/train'
    csv_path = 'data/trainLabels.csv'
    train_csv_path = 'data/train_answer.csv'
    test_csv_path = 'data/test_answer.csv'

    # 하이퍼 파라미터 설정
    num_classes = 10
    batch_size = 32
    epochs = 5
    lr = 1e-3

    # 데이터셋 train test 구분
    split_dataset(csv_path)

    # 텐서로 변경 후 정규화하는 transform 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # training_data 초기화
    training_data = Cifar10Dataset(
        image_dir,
        train_csv_path,
        transform=transform
    )

    # test_data 초기화
    test_data = Cifar10Dataset(
        image_dir,
        test_csv_path,
        transform=transform
    )

    # train셋과 test셋 데이터로딩
    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0)

    # model 설정
    model = LeNet(num_classes=num_classes).to(device)

    # 오차함수 및 옵티마이저 설정
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # epoch마다 훈련과 검증 실행
    for t in range(epochs):
        print(f'Epoch {t+1}\n---------------------------------------')
        train_one_epoch(train_dataloader, device, model, loss_fn, optimizer)
        val_one_epoch(test_dataloader, device, model, loss_fn)
    print("DONE!")

    # 저장
    torch.save(model.state_dict(), 'cifar-net-lenet.pth')
    print('Saved PyTorch model State to cifar-net-lenet.pth')

if __name__== '__main__':
    train(device=args.device)