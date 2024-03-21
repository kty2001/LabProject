import os
import glob
import shutil
from pathlib import Path
import time

import cv2


# 데이터 resize
def data_resize(data_dir):
    data = glob.glob(os.path.join(data_dir + '/*/*.jpg'))

    for index in range(len(data)):
        image_path = data[index]

        # 이미지를 읽어옴
        image = cv2.imread(image_path)

        # (224, 224)로 이미지 크기 조정
        width = 224
        height = 224
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        # 조정한 이미지로 원본 이미지 덮어씌우기
        cv2.imwrite(image_path, resized_image)
        image = cv2.imread(image_path)

# 데이터 train/test로 구분
def data_split(data_dir, split_frequency_rate):
    data = glob.glob(os.path.join(data_dir + '/*/*.jpg'))
    os.makedirs('./train', exist_ok=True)
    os.makedirs('./test', exist_ok=True)

    for index in range(len(data)):
        image_path = data[index]
        image_path_part = Path(data[index]).parts[:]

        os.makedirs('./train/' + image_path_part[-2], exist_ok=True)
        os.makedirs('./test/' + image_path_part[-2], exist_ok=True)

        if index % split_frequency_rate == 0:
            new_image_path = os.path.join(data_dir, 'test', image_path_part[-2])
            shutil.move(image_path, new_image_path)
        else:
            new_image_path = os.path.join(data_dir, 'train', image_path_part[-2])
            shutil.move(image_path, new_image_path)


# 전처리 데이터 설정
data_dir = './'
split_frequency_rate = 10

print('data_resize')
data_resize(data_dir)
print('End\n')

time.sleep(1)

print('data_split')
data_split(data_dir, split_frequency_rate)
print('Done')