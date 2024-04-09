import argparse
import os
import random
import shutil
import json
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

from src.dataset import MyDataset


parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cpu', help='학습에 사용되는 장치')
args = parser.parse_args()


# 데이터 예측 시각화
def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, conf_thr: float = 0.12, n_images: int = 10) -> None:
    # 디렉토리 없으면 생성, 있으면 삭제 후 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    # 클래스명 설정
    classes = ['baguette', 'croissant', 'toast', 'iceamericano', 'powerade']

    # 모델 상태 설정
    model.eval()

    # n_image개의 이미지에 bbox 생성
    indices = random.choices(range(len(testset)), k=n_images)
    for i in tqdm(indices):
        # testset[i] 이미지 데이터 저장 및 예측
        image, _, image_id = testset[i]
        image = [image.to(device)]
        pred = model(image)

        # 이미지 및 예측 값 변환
        image = image[0].detach().cpu().numpy().transpose(1, 2, 0)
        image = np.clip(image, 0, 1)  # 이미지의 픽셀 값을 0에서 1 사이로 클리핑
        pred = {k: v.detach().cpu() for k, v in pred[0].items()}

        plt.imshow(image)   # 이미지 열기
        ax = plt.gca()      # 축 생성

        # 이미지에 박스 생성
        for box, category_id, score in zip(*pred.values()):
            if score >= conf_thr:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                category_id = category_id.item()

                # 박스 생성
                rect = patches.Rectangle(
                    (x1, y1),
                    w, h,
                    linewidth=1,
                    edgecolor='green',
                    facecolor='none'
                )

                # 박스 추가
                ax.add_patch(rect)

                # 텍스트 추가
                ax.text(
                    x1, y1,
                    f'{classes[category_id-1]}: {score:.2f}',
                    c='white',
                    size=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground='green')],
                    family='sans-serif',
                    weight='semibold',
                    va='top', ha='left',
                    bbox=dict(
                        boxstyle='round',
                        ec='green',
                        fc='green',
                    )
                )

        plt.axis('off') # 축 제거
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)  # figure 저장
        plt.clf()       # figure 초기화

# 모델 테스트
def test(device):
    # 디렉토리 설정
    train_image_dir = '.\\images'

    # 파라미터 설정
    num_classes = 5

    with open('json_test.json', 'r') as f:
        json_load = json.load(f)

    # 데이터셋 초기화
    test_data = MyDataset(
        image_dir=train_image_dir,
        json_data=json_load,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
    )

    # 모델 초기화 및 로드
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes+1)
    model.load_state_dict(torch.load('faster-rcnn_pp.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/faster-rcnn')
    print('Saved in ./examples/faster-rcnn')

if __name__ == '__main__':
    test(args.device)