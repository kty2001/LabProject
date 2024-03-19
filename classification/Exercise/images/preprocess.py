import os
import random
import glob

import cv2


data_dir = './'
data = glob.glob(os.path.join(data_dir + '/*/*.jpg'))

for index in range(len(data)):
    image_path = data[index]

    # 이미지를 읽어옴
    image = cv2.imread(image_path)
    print(f"원본 크기: {image.shape[:3]}", end=" / ")

    # (224, 224)로 이미지를 조정
    width = 224
    height = 224
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # 조정한 이미지로 원본 이미지 덮어씌우기
    cv2.imwrite(image_path, resized_image)

    print(f"현재 이미지 크기: {image.shape[:3]}")