import os

from PIL import Image
import pandas as pd

from torch.utils.data import Dataset

from src.utils import CLASSES

class Cifar10Dataset(Dataset):
    def __init__(self, image_dir,label_path, transform):    # 변수 초기화
        super().__init__()

        self.image_dir = image_dir  # 디렉토리
        self.labels = pd.read_csv(label_path)   # csv 파일에서 레이블 읽기
        self.transform = transform  # ?

    def __len__(self):      # Cifar10Dataset 길이 반환
        return len(self.labels) # 레이블 길이 = 데이터셋 길이
    
    def __getitem__(self, index):   # Cifat10Dataset 아이템 반환
        image_id = self.labels.loc[index]   # 레이블에서 index에 맞는 정보 묶어서 가져오기
        image = Image.open(os.path.join(self.image_dir, f"{image_id['id']}.png")).convert('RGB') # RGB로 변환하여 이미지 파일 가져오기
        label = CLASSES.index(image_id['label'])    # 레이블에 맞는 클래스명 저장

        if self.transform is not None:
            image = self.transform(image)   # ?

        return image, label # 이미지와 레이블 반환