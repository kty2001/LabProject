import os
import glob
import shutil
from pathlib import Path
import time


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
data_dir = '.\\'
split_frequency_rate = 10

print('data_split')
data_split(data_dir, split_frequency_rate)
print('Done')