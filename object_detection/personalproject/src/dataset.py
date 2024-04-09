from ast import literal_eval
import os
import glob
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset


# 데이터셋
class MyDataset(Dataset):
    """Wheat 데이터셋 사용자 정의 클래스 정의"""
    def __init__(
            self,
            image_dir:  os.PathLike,
            json_data: dict,
            transform: Optional[Sequence[Callable]] = None,
    ) -> None:
        super().__init__()

        self.image_dir = image_dir  # 이미지 디렉토리 초기화
        self.images = glob.glob(os.path.join(self.image_dir + f"\\*\\*.jpg"))
        self.json_data = json_data
        self.transform = transform

    # 데이터셋 길이 반환
    def __len__(self) -> int:
        return len(self.json_data['images'])  # 이미지 개수
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        idx_image = self.json_data['images'][index]    # index에 맞는 image_id 가져오기

        image = Image.open(os.path.join(idx_image['file_name'])).convert('RGB')      # 이미지 RGB로 변환하여 열기

        boxes = []
        labels = []
        for annotataion in self.json_data['annotations']:
            if annotataion['image_id'] == idx_image['id']:
                boxes.append(annotataion['bbox'])                
                labels.append(annotataion['category_id'])
        
        boxes = np.array(boxes)     # 정답 박스들 numpy형으로 저장

        # 박스의 길이를 위치로 변환
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]     # boxes[, 0]은 박스의 x 시작위치, boxes[, 2]은 박스의 x 길이
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]     # boxes[, 1]은 박스의 y 시작위치, boxes[, 3]은 박스의 y 길이

        # return 값 조정
        if self.transform is not None:
            # bbox 좌표 resize
            origin_w, origin_h = image.size
            w_ratio, h_ratio = 256 / origin_w, 256 / origin_h
            boxes[:, 0] *= w_ratio
            boxes[:, 1] *= h_ratio
            boxes[:, 2] *= w_ratio
            boxes[:, 3] *= h_ratio

            # image에 transform 적용
            image = self.transform(image)       

            # target 텐서화
            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64)
            }

        return image, target, idx_image['id']

# batch 전치
def collate_fn(batch: Tensor) -> Tuple:
    return tuple(zip(*batch))   # batch 전치하여 튜플로 변환