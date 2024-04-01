import argparse
import os
import random
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from src.dataset import collate_fn, WheatDataset
from src.utils import split_dataset, MeanAveragePrecision


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()

# 데이터셋 샘플 시각화
def visualize_dataset(image_dir: os.PathLike, csv_path: os.PathLike, save_dir: os.PathLike, n_images: int = 10) -> None:
    """데이터셋 샘플 bbox 그려서 시각화
    
    :param save_dir: bbox 그린 그림 저장할 폴더 경로
    :type save_dir: os.PathLike
    """
    # 디렉토리 없으면 생성, 있으면 제거 후 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    # 데이터셋 초기화
    dataset = WheatDataset(
        image_dir= image_dir,
        csv_path=csv_path,
        transform=transforms.ToTensor()
    )

    # 데이터셋 범위에서 n_images개 랜덤으로 뽑기
    indices = random.choices(range(len(dataset)), k=n_images)
    for i in indices:
        image, target, image_id = dataset[i]        # dataset[i]의 정보 이동
        image = image.numpy().transpose(1, 2, 0)    # image 차원 변경

        plt.imshow(image)   # 이미지 열기
        ax = plt.gca()      # 그래프 축 가져오기

        # 이미지에 박스 그리기
        for x1, y1, x2, y2 in target['boxes']:
            # 높이 및 길이 계산
            w = x2 - x1
            h = y2 - y1

            # 카테고리 id 설정
            category_id = 'wheat'

            # 직사각형 객체 생성
            rect = patches.Rectangle(
                (x1, y1),           # 왼쪽 하단 모서리 좌표
                w, h,               # 길이 및 높이
                linewidth=1,        # 테두리 두께
                edgecolor='green',  # 직사각형 테두리 색
                facecolor='none'    # 직사각형 내부 색
            )

            # 축에 rect 추가
            ax.add_patch(rect)
            ax.text(
                x1, y1,                 # 텍스트의 왼쪽 하단 모서리 좌표
                category_id,            # 텍스트 내용
                c='white',              # 텍스트 색상
                size=5,                 # 텍스트 크기
                path_effects=[pe.withStroke(linewidth=2, foreground='green')],  # 텍스트 효과
                family='sans-serif',    # 텍스트 폰트
                weight='semibold',      # 텍스트 굵기
                va='top', ha='left',    # 텍스트 정렬
                bbox=dict(              # 텍스트 주변 상자 설정
                    boxstyle='round',   # 둥근 모서리
                    ec='green',         # 테두리 색
                    fc='green',         # 내부 색
                )
            )
        plt.axis('off')     # 축 제거
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)  # 이미지 파일 저장
        plt.clf()   # 활성된 figure 지우고 비우기

# 에포크 훈련
def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Wheat 데이터셋으로 뉴럴 네트워크를 훈련합니다.
    
    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets, _) in enumerate(dataloader):
        # gpu로 이동
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # loss 계산
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 진행도 시각화
        if batch % 10 == 0:
            current = batch * len(images)
            message = 'total loss: {:>4f}, cls loss: {:>4f}, box loss: {:>4f}, obj loss: {:>4f}, rpn loss: {:>4f} [{:>5d}/{:>5d}]'
            message = message.format(
                loss,
                loss_dict['loss_classifier'],
                loss_dict['loss_box_reg'],
                loss_dict['loss_objectness'],
                loss_dict['loss_rpn_box_reg'],
                current,
                size
            )
            print(message)

# 에포크 검증
def val_one_epoch(dataloader: DataLoader, device, model: nn.Module, metric) -> None:
    """CIFAR-10 데이터셋으로 뉴럴 네트워크의 성능을 테스트합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 테스트에 사용되는 장치
    :type device: _device
    :param model: 테스트에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 테스트에 사용되는 오차 함수
    :type loss_fn: nn.Module
    """
    # 값 초기화
    num_batches = len(dataloader)
    test_loss = 0
    test_cls_loss = 0
    test_box_loss = 0
    test_obj_loss = 0
    test_rpn_loss = 0

    # loss 계산
    with torch.no_grad():
        for images, targets, image_ids in dataloader:
            # gpu로 이동
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # loss 계산
            model.train()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # loss 합산
            test_loss += loss
            test_cls_loss += loss_dict['loss_classifier']
            test_box_loss += loss_dict['loss_box_reg']
            test_obj_loss += loss_dict['loss_objectness']
            test_rpn_loss += loss_dict['loss_rpn_box_reg']

            # 예측
            model.eval()
            preds = model(images)

            # metric 업데이트
            metric.update(preds, image_ids)

    # loss 평균 계산 및 출력
    test_loss /= num_batches
    test_cls_loss /= num_batches
    test_box_loss /= num_batches
    test_obj_loss /= num_batches
    test_rpn_loss /= num_batches
    print(f'Test Error: \n Avg loss: {test_loss:>8f} \n Class loss: {test_cls_loss:>8f} \n Box loss: {test_box_loss:>8f} \n Obj loss: {test_obj_loss:>8f} \n RPN loss: {test_rpn_loss:>8f} \n')
    
    # metric 값 계산 및 초기화
    metric.compute()
    metric.reset()
    print()


# 모델 훈련
def train(device) -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    # 디렉토리 설정
    csv_path = 'data/global-wheat-detection/train.csv'
    train_image_dir = 'data/global-wheat-detection/train'
    train_csv_path = 'data/global-wheat-detection/train_answer.csv'
    test_csv_path = 'data/global-wheat-detection/test_answer.csv'

    # 하이퍼파라미터 설정
    num_classes = 1
    batch_size = 16
    epochs = 5
    lr = 1e-3

    # 데이터 분리
    split_dataset(csv_path)

    # 데이터셋 시각화
    visualize_dataset(train_image_dir, train_csv_path, save_dir='examples/global-wheat-detection/train')
    visualize_dataset(train_image_dir, test_csv_path, save_dir='examples/global-wheat-detection/test')

    # 데이터셋 초기화
    training_data = WheatDataset(
        image_dir=train_image_dir,
        csv_path=train_csv_path,
        transform=transforms.ToTensor()
    )
    test_data = WheatDataset(
        image_dir=train_image_dir,
        csv_path=test_csv_path,
        transform=transforms.ToTensor()
    )

    # 데이터로더 초기화
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)

    # 모델 초기화
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes+1).to(device)

    # 옵티마이저 및 평균 정밀도 계산 객체 초기화
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    metric = MeanAveragePrecision(csv_path=test_csv_path)

    # 에포크마다 훈련 및 평가
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_one_epoch(train_dataloader, device, model, optimizer)
        val_one_epoch(test_dataloader, device, model, metric)
    print('Done!')

    # 모델 가중치 저장
    torch.save(model.state_dict(), 'wheat-faster-rcnn.pth')
    print('Saved PtTorch Model State to wheat-faster-rcnn.pth')

if __name__ == '__main__':
    train(args.device)