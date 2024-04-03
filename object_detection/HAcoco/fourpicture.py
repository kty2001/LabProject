import onepicture as op

import os
import json
import random

from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import cv2

import numpy as np
import pandas as pd


def image_transform(img_id):
    image = cv2.imread(f"./examples/{img_id}.jpg")
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def image_concatenate(images):
    image1 = np.concatenate((images[0], images[1]), axis=1)
    image2 = np.concatenate((images[2], images[3]), axis=1)
    image3 = np.concatenate((image1, image2), axis=0)
    print("image concatenate is Done!")

    plt.imshow(image3)
    plt.axis('off')
    plt.savefig(os.path.join("./examples/bigimage.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)  # 이미지 파일 저장
    plt.show()
    plt.clf()   # 활성된 figure 지우고 비우기



import onepicture as op

coco = COCO('./annotations/instances_val2017.json')

cat_ids = coco.getCatIds()
categories = coco.loadCats(cat_ids)
cate_dict = op.makecategorydict(categories)

img_ids = coco.getImgIds()
idxes = random.sample(range(len(img_ids)), 4)

images = [0 for _ in range(len(idxes))]

for i, idx in enumerate(idxes):
    print(f"{i}: {idx} -------------------------")
    img_id, image = op.image_open(img_ids, idx)

    img_annotations = coco.getAnnIds(imgIds = img_ids[idx])
    img_annos = coco.loadAnns(img_annotations)
    
    op.bbox_visualize(img_annos, cate_dict, img_id, image)

    images[i] = image_transform(img_id)

image_concatenate(images)