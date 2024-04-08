from ast import literal_eval
from collections import defaultdict
import json
import os
import random
import glob

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def data_split(images_dir: os.PathLike, load_json: dict, split_rate: float = 0.2) -> None:
    items = os.listdir(images_dir)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(images_dir, item))]
    
    for subdirectory in subdirectories:

        images = glob.glob(os.path.join(f"{images_dir}\\{subdirectory}\\*.jpg"))

        indices = list(range(len(images)))
        random.shuffle(indices)
        split_point = int(split_rate * len(images))

        test_ids = indices[:split_point]

        for test_id in test_ids:
            for img in load_json['images']:
                if img['file_name'] == f'.\\images\\{subdirectory}\\{test_id}.jpg':
                    img['license'] = 1

    train_json = {}
    train_json['info'] = load_json['info']
    train_json['licenses'] = load_json['licenses']
    train_json['annotations'] = load_json['annotations']
    train_json['categories'] = load_json['categories']
    train_json['images'] = []
    test_json = {}
    test_json['info'] = load_json['info']
    test_json['licenses'] = load_json['licenses']
    test_json['annotations'] = load_json['annotations']
    test_json['categories'] = load_json['categories']
    test_json['images'] = []
    for load_json_image in load_json['images']:
        if load_json_image['license'] == 1:
            test_json['images'].append(load_json_image)
        else:
            train_json['images'].append(load_json_image)

    with open('json_train.json', 'w') as make_file:
        json.dump(train_json, make_file)
    with open('json_test.json', 'w') as make_file:
        json.dump(test_json, make_file)
    print("train_json_images_len:", len(train_json['images']))
    print("test_json_images_len:", len(test_json['images']))

# 평균 정밀도 계산
class MeanAveragePrecision:
    def __init__(self, json_path) -> None:
        self.json_path = json_path
        self.detections = []        # 예측 데이터 리스트
        self.coco_gt = COCO(self.json_path)
    
    def update(self, preds, image_ids):
        # 주어진 예측과 이미지 ID에 대해 반복문 실행
        for p, image_id in zip(preds, image_ids):
            # 예측 박스 데이터 변환
            p['boxes'][:, 2] = p['boxes'][:, 2] - p['boxes'][:, 0]      # x1 좌표, x2 좌표 -> x1 좌표, x 길이
            p['boxes'][:, 3] = p['boxes'][:, 3] - p['boxes'][:, 1]      # y1 좌표, y2 좌표 -> y1 좌표, y 길이

            # cpu로 이동 후 numpy로 변환
            p['boxes'] = p['boxes'].cpu().numpy()
            p['scores'] = p['scores'].cpu().numpy()
            p['labels'] = p['labels'].cpu().numpy()

            # image_id를 coco 형식으로 변환해 detection 리스트에 추가
            for b, l, s in zip(*p.values()):    # boxes, labels, scores
                self.detections.append({
                    'image_id': image_id,
                    'category_id': l,
                    'bbox': b.tolist(),
                    'score': s
                })
        print(image_ids)

        print(len(self.detections))

    # 결과 계산
    def compute(self):
        coco_dt = self.coco_gt.loadRes(self.detections)         # detections를 coco형식으로 변환하여 저장

        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')     # COCOeval 객채 생성
        coco_eval.evaluate()        # 예측 결과 평가
        coco_eval.accumulate()      # 예측 결과 누적 값 계산
        coco_eval.summarize()       # 성능 요약 및 출력
        
    # detections 초기화
    def reset(self):
        self.detections = []
