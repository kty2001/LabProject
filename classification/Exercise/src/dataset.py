import os
import random
import glob

from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
import torchvision
import torch

import numpy as np


class ExerciseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 데이터 클래스 저장
        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)     # 리스트로 저장
        try:
            self.classes.remove(".gitkeep")         # .gitkeep 제거
            self.classes.remove("BSCrawling.py")    # BSCrawling.py 제거
            self.classes.remove("preprocess.py")    # preprocess.py 제거
        except:
            pass            
        
        # 데이터 클래스 레이블링
        self.classes_dic = {}
        for label, cls in enumerate(self.classes):
            self.classes_dic[cls] = label       # classes에 저장된 순서로 레이블 매김
        # print(classes_dic)                    # {'풀업': 0, '스쿼트': 1, '벤치프레스': 2, '데드리프트': 3, '오버헤드프레스': 4}
        # print(classes)                        # ['풀업', '스쿼트', '벤치프레스', '데드리프트', '오버헤드프레스']
        
        # 데이터 불러오기
        self.data = glob.glob(os.path.join(data_dir + '/*/*.jpg'))  # 모든 이미지 경로 저장

        # transform 초기화
        self.transform = transform

    def __len__(self):
        return len(self.data)   # data 길이 반환: 총 이미지 수(현재 600)
    
    def __getitem__(self, index):
        image = self.data[index]                            # index에 맞는 이미지 가져오기
        label = self.classes_dic[Path(image).parts[-2]]     # 이미지 디렉토리명의 레이블 저장
        
        image = Image.open(image)               # 이미지 PIL로 열기
        image = np.array(image)                 # 이미지 numpy 배열로 변환

        image = torch.FloatTensor(image)        # 이미지 float형 텐서로 변환
        image = image.permute(2, 0, 1)
        label = torch.LongTensor([int(label)])  # 레이블 long형 텐서로 변환
        return image, label