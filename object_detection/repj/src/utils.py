from ast import literal_eval
from collections import defaultdict
import json
import os
import glob
import random

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 데이터셋 필터링
def cleaning_dataset(json_data: dict):
    print("Cleaning Dataset")

    for image in json_data['images']:
        bboxes = []
        for anno in json_data['annotations']:
            if image['id'] == anno['image_id']:
                bboxes.append(anno)
        if len(bboxes) == 0:
            image['license'] = -1

    print("Cleaning Done")


# 데이터셋 분리
def split_dataset(image_path: os.PathLike, json_data: dict, split_rate: float = 0.2) -> None:
    print("Split Dataset")

    images = glob.glob(os.path.join(f"{image_path}\\*.jpg"))

    indices = list(range(len(images)))
    random.shuffle(indices)
    split_point = int(split_rate * len(images))

    test_ids = indices[:split_point]

    for img in json_data['images']:
        if img['license'] == -1:
            continue
        for test_id in test_ids:
            if img['file_name'] == images[test_id].split("\\")[-1]:
                img['license'] = 0

    train_json = {}
    train_json['categories'] = json_data['categories']
    train_json['images'] = []
    train_json['annotations'] = []

    test_json = {}
    test_json['categories'] = json_data['categories']
    test_json['images'] = []
    test_json['annotations'] = []


    for json_data_image in json_data['images']:
        if json_data_image['license'] == -1:
            continue

        elif json_data_image['license'] == 0:
            test_json['images'].append(json_data_image)
            for json_data_anno in json_data['annotations']:
                if json_data_anno['image_id'] == json_data_image['id']:
                    test_json['annotations'].append(json_data_anno)
        else:
            train_json['images'].append(json_data_image)
            for json_data_anno in json_data['annotations']:
                if json_data_anno['image_id'] == json_data_image['id']:
                    train_json['annotations'].append(json_data_anno)

    with open('train_json.json', 'w') as make_file:
        json.dump(train_json, make_file)
    with open('test_json.json', 'w') as make_file:
        json.dump(test_json, make_file)
    print("train_json_images_len:", len(train_json['images']))
    print("test_json_images_len:", len(test_json['images']))
    
    print("Split Done")


# 평균 정밀도 계산
class MeanAveragePrecision:
    def __init__(self, json_path: os.PathLike, json_data: dict, change_size: int) -> None:
        self.json_data = json_data
        self.json_path = json_path
        self.change_size = change_size
        self.coco_gt = COCO(json_path)

        self.detections = []
    
    def update(self, preds, image_ids):
        # 주어진 예측과 이미지 ID에 대해 반복문 실행
        for p, image_id in zip(preds, image_ids):
            # 예측 박스 데이터 변환
            p['boxes'][:, 2] = p['boxes'][:, 2] - p['boxes'][:, 0]
            p['boxes'][:, 3] = p['boxes'][:, 3] - p['boxes'][:, 1]

            # cpu로 이동 후 numpy로 변환
            p['boxes'] = p['boxes'].cpu().numpy()
            p['scores'] = p['scores'].cpu().numpy()
            p['labels'] = p['labels'].cpu().numpy()

            # # 원본 이미지와의 비율 적용
            for img in self.json_data['images']:
                if img['id'] == image_id:
                    origin_w, origin_h = img['width'], img['height']
                    break
            
            w_ratio = origin_w / self.change_size
            h_ratio = origin_h / self.change_size

            p['boxes'][:, 0] *= w_ratio
            p['boxes'][:, 1] *= h_ratio
            p['boxes'][:, 2] *= w_ratio
            p['boxes'][:, 3] *= h_ratio

            # image_id를 coco 형식으로 변환해 detection 리스트에 추가
            for b, l, s in zip(*p.values()):
                self.detections.append({
                    'image_id': image_id,
                    'category_id': int(l),
                    'bbox': b.tolist(),
                    'score': float(s)
                })

    # 결과 계산
    def compute(self):

        copy_detection = self.detections.copy()

        with open('detections.json', 'w') as make_file:
            json.dump(copy_detection, make_file, indent=4, ensure_ascii=False)

        dt = []
        for detec in copy_detection:
            if detec['score'] >= 0.1:
                dt.append(detec)

        with open('dt.json', 'w') as make_file:
            json.dump(dt, make_file, indent=4, ensure_ascii=False)

        try:
            coco_dt = self.coco_gt.loadRes(self.detections)

            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        except:
            print("\nNo detections")

        print(f"\ndetections len: {len(copy_detection)}, dt len: {len(dt)}\n")
        
    # detections 초기화
    def reset(self):
        self.detections = []
