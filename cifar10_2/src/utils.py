import os
import random

import pandas as pd


CLASSES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

def split_dataset(csv_path: os.PathLike, split_rate: float = 0.2) -> None:
    root_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    size = len(df)
    indices = list(range(size))

    random.shuffle(indices)

    split_point = int(split_rate * size)

    test_ids = indices[:split_point]
    train_ids = indices[split_point:]

    test_df = df.loc[test_ids]
    test_df.to_csv(os.path.join(root_dir, "test_answer.csv"), index=False)
    train_df = df.loc[test_ids]
    train_df.to_csv(os.path.join(root_dir, "train_answer.csv"), index=False)