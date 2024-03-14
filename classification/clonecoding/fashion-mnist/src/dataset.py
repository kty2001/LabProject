import os

import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset


# FashionMnistDataset 클래스 생성
class FashionMnistDataset(Dataset):
    def __init__(self, image_dir, label_path):
        super().__init__()

        # 변수 초기화
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_path)

    # __len__ 설정
    def __len__(self):
        return len(self.labels)
    
    # __getitem 설정
    def __getitem__(self, index):
        image_id = self.labels.loc[index] # index에 맞는 행 불러오기
        image = Image.open(os.path.join(self.image_dir, f"{image_id['id']}.jpg")) # image_id에 맞는 이미지 image 디렉토리에서 불러오기
        label = image_id['label'] # image_id의 label 받기

        # 이미지 넘파이 배열로 변환
        image = np.array(image)

        # 변수형 변경
        image = torch.FloatTensor(image)
        label = torch.LongTensor([int(label)])

        # image, label 반환
        return image, label