import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.dataset import FashionMnistDataset
from src.model import NeuralNetwork
from src.utils import split_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()

# 훈련 에포크 함수
def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """FashionMNIST 데이터셋으로 뉴럴 네트워크를 훈련
    
    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    """

    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        # 데이터 gpu로 이동
        images = images.to(device)
        targets = targets.to(device)
        targets = torch.flatten(targets)

        # 예측 및 오차 계산
        preds = model(images)
        loss = loss_fn(preds, targets)
        
        # 역전파 진행
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 진행상황 및 loss 출력
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

# 검증 에포크 진행
def val_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module) -> None:
    """FashionMNIST 데이터셋으로 뉴럴 네트워크의 성능을 테스트합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    # 기울기 계산 생략하여 진행
    with torch.no_grad():
        for images, targets in dataloader:
            # 데이터 gpu로 이동
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.flatten(targets)

            # 예측
            preds = model(images)

            # test_loss 및 correct 합산
            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).float().sum().item()
    # 크기에 맞춰 나누어 평균 계산 및 출력
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

# 훈련 진행 함수
def train(device):
    image_dir = 'data/images'
    csv_path = 'data/answers.csv'
    train_csv_path = 'data/train_answer.csv'
    test_csv_path = 'data/test_answer.csv'

    num_classes = 10
    batch_size = 32
    epochs = 5
    lr = 1e-3

    # csv_path 데이터 train과 test로 분류
    split_dataset(csv_path)

    # 데이터셋 생성
    training_data = FashionMnistDataset(
        image_dir,
        train_csv_path,
    )
    test_data = FashionMnistDataset(
        image_dir,
        test_csv_path,
    )

    # 데이터로더 생성
    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0)

    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 초기화 및 gpu로 이동
    model = NeuralNetwork(num_classes=num_classes).to(device)

    # 오차 함수와 옵티마이저 초기화
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 에포크마다 훈련과 검증을 반복
    for t in range(epochs):
        print(torch.cuda.is_available())
        print(f'Epoch {t+1}\n-----------------------')
        train_one_epoch(train_dataloader, device, model, loss_fn, optimizer)
        val_one_epoch(test_dataloader, device, model, loss_fn)
    print('Done!')

    # 결과 저장
    torch.save(model.state_dict(), 'fashion-mnist-net.pth')
    print('Saved PyTorch Model State to fashion-mnist-net.pth')

if __name__ == "__main__":
    train(args.device)