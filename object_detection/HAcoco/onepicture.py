import os
import json
import random

from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from PIL import Image

import numpy as np
import pandas as pd


def bbox_visualize(img_annos, cate_dict, image_id, image):
    
    plt.imshow(image)   # 이미지 열기
    ax = plt.gca()      # 그래프 축 가져오기

    # 이미지에 박스 그리기
    for img_anno in img_annos:
        x1 = img_anno['bbox'][0]
        y1 = img_anno['bbox'][1]
        w = img_anno['bbox'][2]
        h = img_anno['bbox'][3]

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
            cate_dict[img_anno['category_id']],            # 텍스트 내용
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
    plt.savefig(os.path.join("./examples/", f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)  # 이미지 파일 저장
    print(f"Bbox_visualize is Done!\nSave in \"./examples/{image_id}.jpg\"")
    plt.clf()   # 활성된 figure 지우고 비우기


def makecategorydict(categories):
    cate_dict ={}
    for category in categories:
        cate_dict[category['id']] = category['name']

    print("Make_categorydict is Done!")
    return cate_dict
    

def image_open(img_ids, idx = 0):
    img_id = str(img_ids[idx]).zfill(12)
    image_path = os.path.join("./val2017/" + f"{img_id}.jpg")

    # PIL로 이미지 시각화
    # image = np.array(Image.open(image_path))
    # plt.imshow(image)
    # plt.show(image)

    print(f"idx: {idx}, img_id: {img_id}.jpg")
    image = plt.imread(image_path)
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    print("Image_open is Done!")
    return img_id, image

coco = COCO('./annotations/instances_val2017.json')

img_ids = coco.getImgIds()
idx = random.randint(0, len(img_ids))
img_id, image = image_open(img_ids, idx)

img_annotations = coco.getAnnIds(imgIds = img_ids[idx])
img_annos = coco.loadAnns(img_annotations)

cat_ids = coco.getCatIds()
categories = coco.loadCats(cat_ids)
cate_dict = makecategorydict(categories)

bbox_visualize(img_annos, cate_dict, img_id, image)