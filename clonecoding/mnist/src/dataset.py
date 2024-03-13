import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torch

# 클래스 생성
class MnistDataset(Dataset):
    # 변수 초기화
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.data = glob.glob(os.path.join(data_dir + '/*/*.jpg')) # data_dir 내부에 폴더 존재
        print(self.data)
        self.transform = transform

    # __len__ 초기화
    def __len__(self):
        return len(self.data)
    
    # __gititem__ 초기화
    def __getitem__(self, index):
        image = self.data[index]
        # print(image)
        # label = self.data[index].split('\\')[-2]
        label = Path(image).parts[-2] # 레이블에 맞는 데이터 들어있는 폴더 이름
        # print(label)
        image = Image.open(image)
        # print(image, label)
        image = np.array(image)

        image = torch.FloatTensor(image)
        label = torch.LongTensor([int(label)])
        return image, label

if __name__ == "__main__":
    dataset = MnistDataset(data_dir='./data/MNIST - JPG - training', transform=None)
    print(len(dataset))
    print(dataset[0])