from glob import glob
import os
import random
from typing import Sequence

import pandas as pd

from torchvision import transforms

def split_dataset(label_dir: os.PathLike, split_rate: float = 0.2) -> None:
    root_dir = os.path.dirname(label_dir)

    image_ids = []
    for path in glob(os.path.join(label_dir, '*.png')):
        file_name = os.path.split(path)[-1]
        image_id = os.path.splitext(file_name)[0]
        image_ids.append(image_id)

    random.shuffle(image_ids)

    split_point = int(split_rate * len(image_ids))

    test_ids = image_ids[:split_point]
    train_ids = image_ids[split_point:]

    test_df = pd.DataFrame({'image_id': test_ids})
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)
    train_df = pd.DataFrame({'image_id': train_ids})
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)


def get_transform(size: Sequence[int]):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
    ])


def get_mask_transform(size: Sequence[int]):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            size,
            interpolation=transforms.InterpolationMode.NEAREST
        ),
    ])