from ast import literal_eval
from collections import defaultdict
import json
import os

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 데이터셋 분리
def split_dataset(csv_path: os.PathLike, split_rate: float = 0.2) -> None:
    """Dirty-MNIST 데이터셋을 비율에 맞춰 train / test로 나눕니다.
    
    :param path: Dirty-MNIST 데이터셋 경로
    :type path: os.PathLike
    :param split_rate: train과 test로 나누는 비율
    :type split_rate: float
    """
    # 데이터 로드 및 셔플
    root_dir = os.path.dirname(csv_path)            # csv 파일 디렉토리 저장
    df = pd.read_csv(csv_path)                      # csv 파일 pandas로 읽기
    df = df.sample(frac=1).reset_index(drop=True)   # 데이터 셔플 후 새 인덱스 할당 및 기존 인덱스 삭제

    # 데이터 그룹화
    grouped = df.groupby(by='image_id')             # image_id 기준으로 데이터 그룹화
    grouped_list = [group for _, group in grouped]  # 데이터 그룹 리스트화

    # 데이터 분리
    split_point = int(split_rate * len(grouped))    # 그룹 객채의 개수를 기준으로 분리
    test_ids = grouped_list[:split_point]
    train_ids = grouped_list[split_point:]

    # csv에 데이터 저장
    test_df = pd.concat(test_ids)                   # 리스트로 묶인 데이터 프레임 병합
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)
    train_df = pd.concat(train_ids)                 
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)


# 평균 정밀도 계산
class MeanAveragePrecision:
    def __init__(self, csv_path: os.PathLike) -> None:
        self.id_csv2coco = {}       # CSV 파일의 ID와 COCO 형식의 데이터셋에서의 ID 간의 매핑을 저장할 딕셔너리
        json_path = self.to_coco(csv_path)  # csv 파일 coco형식의 json 파일로 변형
        self.coco_gt = COCO(json_path)      # coco 데이터셋 로드

        self.detections = []        # 예측 데이터 리스트

    # 데이터 json파일에 저장
    def to_coco(self, csv_path: os.PathLike) -> os.PathLike:
        df = pd.read_csv(csv_path)      # csv 파일 읽기

        grouped = df.groupby(by='image_id')     # image_id 기준으로 그룹화
        grouped_dict = {image_id: group for image_id, group in grouped}     # 데이터 그룹 딕셔너리화

        res = defaultdict(list)     # 기본값이 빈 리스트인 defaultdict 객체를 생성
        n_id = 0

        # 데이터 res에 저장
        for image_id, (file_name, group) in enumerate(grouped_dict.items()):
            # res에 데이터 추가
            res['images'].append({
                'id': image_id,
                'width': 1024,
                'height': 1024,
                'file_name': f'{file_name}.jpg',
            })
            
            # 파일명을 키로 image_id를 값으로 저장
            self.id_csv2coco[file_name] = image_id

            # res에 애노테이션(정답박스?) 추가
            for _, row in group.iterrows():
                x1, y1, w, h = literal_eval(row['bbox'])
                res['annotations'].append({
                    'id': n_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'area': w * h,
                    'bbox': [x1, y1, w, h],
                    'iscrowd': 0,   # 해당 객체 박스 형태로 정의(1이면 세그멘테이션으로 정의)
                })
                n_id += 1
        
        # res에 카테고리 키 추가
        res['categories'].extend([{'id': 1, 'name': 'wheat'}])

        # 데이터 json파일로 저장
        root_dir = os.path.split(csv_path)[0]   # 파일명과 디렉토리 분리
        save_path = os.path.join(root_dir, "coco_annotations.json") # 저장할 디렉토리 설정
        with open(save_path, 'w') as f:
            json.dump(res, f)       # json 파일에 쓰기

        return save_path
    
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
            image_id = self.id_csv2coco[image_id]
            for b, l, s in zip(*p.values()):    # boxes, labels, scores
                self.detections.append({
                    'image_id': image_id,
                    'category_id': l,
                    'bbox': b.tolist(),
                    'score': s
                })

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
