import os
import random

import pandas as pd

CLASSES = [		# 클래스 이름 지정
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
	""" cifar-10 데이터셋을 비율에 맞춰 train / test로 구분

	param csv_path: cifar-10 데이터셋 경로
	type path: os.PathLike
	param split_rate: train과 test로 데이터 나누는 비율
	type split_rate: float
	"""

	root_dir = os.path.dirname(csv_path)

	df = pd.read_csv(csv_path)
	size = len(df)
	indices = list(range(size))		# size 크기의 리스트 생성

	# 리스트 셔플
	random.shuffle(indices)

	# train test 나눌 포인트 설정
	split_point = int(split_rate * size)

	# train test 나누기
	test_ids = indices[:split_point]
	train_ids = indices[split_point:]

	# train test 정보 loc 이용하여 묶어서 csv 파일 생성
	test_df = df.loc[test_ids]
	test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)
	train_df = df.loc[train_ids]
	train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)