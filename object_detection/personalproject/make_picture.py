import os
import json
import random

from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import cv2

import numpy as np


def image_transform(bbox_image_path):
    trans_image = cv2.imread(bbox_image_path)
    trans_image = cv2.resize(trans_image, (256, 256))
    trans_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR)

    return trans_image


def image_concatenate(images):
    image1 = np.concatenate((images[0], images[1]), axis=1)
    image2 = np.concatenate((images[2], images[3]), axis=1)
    image3 = np.concatenate((image1, image2), axis=0)
    print("image concatenate is Done!")

    plt.imshow(image3)
    plt.axis('off')
    plt.savefig(os.path.join("./examples/bigimage.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)  # 이미지 파일 저장
    plt.show()
    plt.clf()   # 활성된 figure 지우고 비우기


def bbox_visualize(img_annotations, cate_dict, img_filename, image):
    
    plt.imshow(image)   # 이미지 열기
    ax = plt.gca()      # 그래프 축 가져오기

    # 이미지에 박스 그리기
    for img_annotation in img_annotations:
        x1 = img_annotation['bbox'][0]
        y1 = img_annotation['bbox'][1]
        w = img_annotation['bbox'][2]
        h = img_annotation['bbox'][3]

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
            cate_dict[img_annotation['category_id']],            # 텍스트 내용
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
    plt.savefig(os.path.join(".\\examples", os.path.basename(img_filename)), dpi=150, bbox_inches='tight', pad_inches=0)  # 이미지 파일 저장
    print(f"Bbox_visualize is Done!\nSave in \"./examples/{os.path.basename(img_filename)}\"")
    plt.clf()   # 활성된 figure 지우고 비우기


def makecategorydict(categories):
    cate_dict = {}
    for category in categories:
        cate_dict[category['id']] = category['name']

    print("Make_categorydict is Done!")
    return cate_dict


def image_open(images, idx = 0):
    img_id = images[idx]['id']
    img_filename = images[idx]['file_name']
    image_path = os.path.join(img_filename)
    print(f"idx: {idx}, img_filename: {img_filename}")
    
    image = plt.imread(image_path)
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    print("Image_open is Done!")
    return img_id, img_filename, image


def step():
    with open('./labelme2coco.json', 'r') as f:
        coco = json.load(f)

    if not os.path.exists(".\\examples"):
        os.makedirs(".\\examples")

    categories = coco['categories']
    images = coco['images']
    annotations = coco['annotations']
    idxes = random.sample(range(len(images)), 4)
    print("idxes: ", idxes)

    cate_dict = makecategorydict(categories)
    np_images = [0 for _ in range(len(idxes))]

    for i, idx in enumerate(idxes):
        print(f"{i}: {idx} -------------------------")
        img_id, img_filename, image = image_open(images, idx)

        img_annotations = []
        for annotation in annotations:
            if annotation['image_id'] == img_id:
                img_annotations.append(annotation)
        
        bbox_visualize(img_annotations, cate_dict, img_filename, image)
        bbox_image_path = f'.\\examples\\{os.path.basename(img_filename)}'
        np_images[i] = image_transform(bbox_image_path)

    image_concatenate(np_images)

if __name__ == "__main__":
    step()