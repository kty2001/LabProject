import argparse
import os
import random
import shutil
import json
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from src.dataset import collate_fn, MyDataset
from src.utils import data_split, MeanAveragePrecision


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()

# 데이터셋 샘플 시각화
def visualize_dataset(image_dir: os.PathLike, json_data: dict, save_dir: os.PathLike, n_images: int = 10) -> None:
    # 디렉토리 없으면 생성, 있으면 제거 후 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    
    # json 데이터 카테고리 생성
    cate_dict = {}
    for i, category in enumerate(json_data['categories']):
        cate_dict[i] = category['name']

    # 데이터셋 초기화
    dataset = MyDataset(
        image_dir= image_dir,
        json_data=json_data,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    )
    

    # 데이터셋 범위에서 n_images개 랜덤으로 뽑기
    indices = random.choices(range(len(dataset)), k=n_images)
    for i in indices:
        image, target, image_id = dataset[i]        # dataset[i]의 정보 이동
        image = image.numpy().transpose(1, 2, 0)    # image 차원 변경
        image = np.clip(image, 0, 1)  # 이미지의 픽셀 값을 0에서 1 사이로 클리핑

        plt.imshow(image)   # 이미지 열기
        ax = plt.gca()      # 그래프 축 가져오기

        # 이미지에 박스 그리기
        for i, (x1, y1, x2, y2) in enumerate(target['boxes']):
            # 높이 및 길이 계산
            w = x2 - x1
            h = y2 - y1

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
                cate_dict[int(target['labels'][i])],            # 텍스트 내용
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
    
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets, _) in enumerate(dataloader):
        # gpu로 이동
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # loss 계산
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

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
            loss = sum(loss_dict.values())

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

    # 디렉토리 설정
    images_dir = '.\\images'

    # 하이퍼파라미터 설정
    num_classes = 5
    batch_size = 16
    epochs = 100
    lr = 1e-4

    with open('labelme2coco.json', 'r') as f:
        json_load = json.load(f)

    # 데이터 분리
    data_split(images_dir, json_load)

    with open('json_train.json', 'r') as f:
        json_train = json.load(f)

    with open('json_test.json', 'r') as f:
        json_test = json.load(f)
        json_test_path = '.\\json_test.json'

    # 데이터셋 시각화
    visualize_dataset(images_dir, json_train, save_dir='examples/train')
    visualize_dataset(images_dir, json_test, save_dir='examples/test')

    # 데이터셋 초기화
    training_data = MyDataset(
        image_dir=images_dir,
        json_data=json_train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    )
    test_data = MyDataset(
        image_dir=images_dir,
        json_data=json_test,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    )

    # 데이터로더 초기화
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)

    # 모델 초기화
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes+1).to(device)

    # 옵티마이저 및 평균 정밀도 계산 객체 초기화
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    metric = MeanAveragePrecision(json_path=json_test_path)

    # 에포크마다 훈련 및 평가
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_one_epoch(train_dataloader, device, model, optimizer)
        val_one_epoch(test_dataloader, device, model, metric)
    print('Done!')

    # 모델 가중치 저장
    torch.save(model.state_dict(), 'faster-rcnn_pp.pth')
    print('Saved PtTorch Model State to faster-rcnn_pp.pth')

if __name__ == '__main__':
    train(args.device)