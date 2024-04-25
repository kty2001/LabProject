from ast import literal_eval
import os
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset
import time


# 밀 데이터셋
class MyDataset(Dataset):
    def __init__(
            self,
            image_path:  os.PathLike,
            json_data: dict,
            change_size: int,
            transform: Optional[Sequence[Callable]] = None,
    ) -> None:
        super().__init__()

        self.image_path = image_path
        self.json_data = json_data
        self.change_size = change_size
        self.transform = transform

    # 데이터셋 길이 반환
    def __len__(self) -> int:
        return len(self.json_data['images'])  # 이미지 개수
    
    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        idx_image = self.json_data['images'][idx]

        image = Image.open(os.path.join(self.image_path, idx_image['file_name'])).convert('RGB')

        boxes = []
        labels = []
        for anno in self.json_data['annotations']:
            if anno['image_id'] == idx_image['id']:
                boxes.append(anno['bbox'])
                labels.append(anno['category_id'])

        boxes = np.array(boxes)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        cate_list = []
        for cate in self.json_data['categories']:
            cate_list.append(cate['id'])
        
        for i in range(len(labels)):
            labels[i] = cate_list.index(labels[i])

        # return 값 조정
        if self.transform is not None:
            # bbox resize
            # origin_w, origin_h = image.size
            # w_ratio, h_ratio = self.change_size / origin_w, self.change_size / origin_h
            # boxes[:, 0] *= w_ratio
            # boxes[:, 1] *= h_ratio
            # boxes[:, 2] *= w_ratio
            # boxes[:, 3] *= h_ratio

            # image에 transform 적용
            image = np.array(image) / 255.0
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