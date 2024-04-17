import argparse
import os
import random
import shutil
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.dataset import collate_fn, MyDataset
from src.utils import split_dataset, cleaning_dataset, MeanAveragePrecision


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()

# 데이터셋 샘플 시각화
def visualize_dataset(image_path: os.PathLike, json_data: dict, change_size ,save_dir: os.PathLike, n_images: int = 5) -> None:

    # 디렉토리 없으면 생성, 있으면 제거 후 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    # 데이터셋 초기화
    dataset = MyDataset(
        image_path=image_path,
        json_data=json_data,
        change_size=change_size,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((change_size, change_size)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    )

    # json 데이터 카테고리 생성
    cate_dict = {}
    for category in json_data['categories']:
        cate_dict[category['id']] = category['name']

    cate_list = []
    for category in json_data['categories']:
        cate_list.append(category['id'])

    # 데이터셋 범위에서 n_images개 랜덤으로 뽑기
    indices = random.choices(range(len(dataset)), k=n_images)
    indices = [300, 50, 100, 150, 200]
    for i in indices:
        image, target, image_id = dataset[i]        # dataset[i]의 정보 이동
        image = image.numpy().transpose(1, 2, 0)    # image 차원 변경

        plt.imshow(image)   # 이미지 열기
        ax = plt.gca()      # 그래프 축 가져오기

        # 이미지에 박스 그리기
        for i, (x1, y1, x2, y2) in enumerate(target['boxes']):
            # 높이 및 길이 계산
            w = x2 - x1
            h = y2 - y1

            # 카테고리 id 설정
            category_id = cate_list[target['labels'][i]]

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
                cate_dict[category_id],            # 텍스트 내용
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
        images = [torch.tensor(image, dtype=torch.float32).to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # loss 계산        
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 진행도 시각화
        if batch % 20 == 0:
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

    # loss 계산
    with torch.no_grad():
        for images, _, image_ids in dataloader:
            # gpu로 이동
            images = [torch.tensor(image, dtype=torch.float32).to(device) for image in images]
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 예측
            model.eval()
            preds = model(images)

            # metric 업데이트
            metric.update(preds, image_ids)

    # metric 값 계산 및 초기화
    metric.compute()
    metric.reset()
    print()


# 모델 훈련
def train(device) -> None:
    
    # 디렉토리 설정
    image_path = ".\\images\\val2017\\val2017"

    # json data 설정
    with open('instances_val2017.json', 'r') as f:
        json_data = json.load(f)

    # 하이퍼파라미터 설정
    num_classes = len(json_data['categories'])
    batch_size = 4
    epochs = 3
    lr = 1e-4
    change_size = 256

    # 데이터 전처리
    # cleaning_dataset(json_data)
    # split_dataset(image_path, json_data)

    # train/test data 설정
    with open('train_json.json', 'r') as f:
        train_json_data = json.load(f)
    with open('test_json.json', 'r') as f:
        test_json_data = json.load(f)

    # 데이터셋 시각화
    visualize_dataset(image_path, train_json_data, change_size, save_dir='examples/train')
    visualize_dataset(image_path, test_json_data, change_size, save_dir='examples/test')

    # 데이터셋 초기화
    training_data = MyDataset(
        image_path=image_path,
        json_data=train_json_data,
        change_size=change_size,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((change_size, change_size)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    )
    test_data = MyDataset(
        image_path=image_path,
        json_data=test_json_data,
        change_size=change_size,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((change_size, change_size)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    )

    # 데이터로더 초기화
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)

    # 모델 초기화
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    model.to(device)

    # 옵티마이저 및 평균 정밀도 계산 객체 초기화
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    metric = MeanAveragePrecision(json_path=".\\test_json.json", json_data=test_json_data)

    # 에포크마다 훈련 및 평가
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_one_epoch(train_dataloader, device, model, optimizer)
        val_one_epoch(test_dataloader, device, model, metric)
    print('Done!')

    # 모델 가중치 저장
    torch.save(model.state_dict(), 'coco-faster-rcnn.pth')
    print('Saved PtTorch Model State to coco-faster-rcnn.pth')

if __name__ == '__main__':
    train(args.device)