import os
import random

import cv2
import pandas as pd
from tqdm import tqdm # 반복문 등 진행상황 표시


# 데이터 저장
def save_data():
    # 데이터 위치 지정
    train_csv_path = 'data/fashion-mnist_train.csv'
    test_csv_path = 'data/fashion-mnist_test.csv'

    # labels와 images 데이터 저장
    image_id = 0
    all_labels = []
    for csv_path in [train_csv_path, test_csv_path]:
        # 경로 지정
        root_dir = os.path.dirname(csv_path) # 경로 추출: ./data
        os.makedirs(os.path.join(root_dir, 'images'), exist_ok=True)

        # labels 및 images 분류
        data = pd.read_csv(csv_path).to_numpy()
        labels = data[:, 0]
        all_labels.extend(labels)
        images = data[:, 1:]

        # 이미지 디렉토리에 저장
        for image in tqdm(images):
            cv2.imwrite(os.path.join(root_dir, f'images/{image_id}.jpg'), image.reshape(28, 28, 1)) # 지정된 디렉토리에 image_id.jpg 명으로 reshape되어 저장
            image_id += 1
    # id와 label 열 생성하여 answer.csv에 저장
    labels_df = pd.DataFrame(all_labels, columns=['label'])
    labels_df.index.name = 'id'
    labels_df.to_csv(os.path.join(root_dir, 'answer.csv'), index=True)

# 데이터셋 나누기
def split_dataset(csv_path: os.PathLike, split_rate: float = 0.2) -> None:
    """Dirty-MNIST 데이터셋을 비율에 맞춰 train / test로 구분
    
    param path: Dirty-MNIST 데이터셋 경로
    type path: os.PathLike
    param split_rate: train과 test로 데이터 나누는 비율
    type split_rate: float
    """
    # 경로 지정
    root_dir = os.path.dirname(csv_path) # 경로 추출

    # df, size 지정 및 indices 생성
    df = pd.read_csv(csv_path)
    size = len(df)
    indices = list(range(size))

    # indices 셔플
    random.shuffle(indices)

    # split_rate에 맞춰 split_point 지정
    split_point = int(split_rate * size)

    # split_point에 맞춰 test, train 셋 지정
    test_ids = indices[:split_point]
    train_ids = indices[split_point:]

    # test_ids, train_ids에 해당하는 행 선택하여 각각의 csv 생성
    test_df = df.loc[test_ids]
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)
    train_df = df.loc[train_ids]
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)
    
if __name__ == "__main__":
    save_data()