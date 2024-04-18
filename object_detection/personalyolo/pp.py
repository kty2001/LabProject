from ultralytics.data.converter import convert_coco
import json
import os
import glob
import random
import shutil


def change_image_structure(images_path, move_path):

    if os.path.exists(move_path):
        shutil.rmtree(move_path)
    os.makedirs(move_path)
    
    dataset = glob.glob(os.path.join(f"{images_path}\\*\\*.jpg"))
    print("len(dataset):", len(dataset))
    for data in dataset:
        image_path = data.split("\\")
        new_image_name = image_path[-2] + image_path[-1]
        new_image_path = os.path.join(move_path, new_image_name)
        shutil.copy(data, new_image_path)

        json_path = data.replace("jpg", "json").split("\\")
        new_json_name = json_path[-2] + json_path[-1]
        new_json_path = os.path.join(move_path, new_json_name)
        shutil.copy(data.replace("jpg", "json"), new_json_path)


def labelme2coco(data_dir):
    data_dir = data_dir
    data = glob.glob(os.path.join(data_dir + '/*.json'))
    print("data len:", len(data))

    make_coco = {'images': [], 'annotations': [], 'categories': []}

    for img_id in range(len(data)):

        with open(data[img_id], 'r') as f:
            load_json = json.load(f)

        make_coco['images'].append({'license': 1, 'file_name': os.path.basename(data[img_id]).replace('json','jpg'), 'height': load_json['imageHeight'], 'width': load_json['imageWidth'], "id": img_id})

        for box_id, box in enumerate(load_json['shapes']):
            category_id_list = [category['name'] for category in make_coco['categories']]
            if box['label'] not in category_id_list:
                make_coco['categories'].append({'id': len(category_id_list), 'name': box['label']})
                category_id_list = [category['name'] for category in make_coco['categories']]

            x1, y1 = min(box['points'][0][0], box['points'][1][0]), min(box['points'][0][1], box['points'][1][1])
            x2, y2 = max(box['points'][0][0], box['points'][1][0]), max(box['points'][0][1], box['points'][1][1])
            w, h = x2 - x1, y2 - y1
            make_coco['annotations'].append({'id': box_id, 'image_id': img_id, 'category_id': category_id_list.index(box['label']), 'area': w * h, 'bbox': [x1, y1, w, h], 'iscrowd': 0})

    with open('labelme2coco.json', 'w') as make_file:
        json.dump(make_coco, make_file)


def cleaning_dataset(json_data: dict):
    print("Cleaning Dataset")

    remove_image = 0
    for image in json_data['images']:
        bboxes = []
        for anno in json_data['annotations']:
            if image['id'] == anno['image_id']:
                bboxes.append(anno)
        if len(bboxes) == 0:
            image['license'] = -1
            remove_image += 1
    print("removed images:", remove_image)
    
    print("Cleaning Done")


def split_dataset(images_path: os.PathLike, json_data: dict, split_rate: float = 0.2) -> None:
    print("Split Dataset")

    images = glob.glob(os.path.join(f"{images_path}\\*.jpg"))

    indices = list(range(len(images)))
    random.shuffle(indices)
    split_point = int(split_rate * len(images))

    test_ids = indices[:split_point]

    for img in json_data['images']:
        if img['license'] == -1:
            continue
        for test_id in test_ids:
            if img['file_name'] == images[test_id].split("\\")[-1]:
                img['license'] = 0

    train_json = {}
    train_json['categories'] = json_data['categories']
    train_json['images'] = []
    train_json['annotations'] = []

    test_json = {}
    test_json['categories'] = json_data['categories']
    test_json['images'] = []
    test_json['annotations'] = []


    for json_data_image in json_data['images']:
        if json_data_image['license'] == -1:
            continue

        elif json_data_image['license'] == 0:
            test_json['images'].append(json_data_image)
            for json_data_anno in json_data['annotations']:
                if json_data_anno['image_id'] == json_data_image['id']:
                    test_json['annotations'].append(json_data_anno)
        else:
            train_json['images'].append(json_data_image)
            for json_data_anno in json_data['annotations']:
                if json_data_anno['image_id'] == json_data_image['id']:
                    train_json['annotations'].append(json_data_anno)

    with open('.\\annotations\\train.json', 'w') as make_file:
        json.dump(train_json, make_file)
    with open('.\\annotations\\test.json', 'w') as make_file:
        json.dump(test_json, make_file)
    print("train_json_images_len:", len(train_json['images']))
    print("test_json_images_len:", len(test_json['images']))
    
    print("Split Done")


def manual_convert(images_path, yolo_path):
    train_image_dir = f".\\{yolo_path}\\images\\train"
    if os.path.exists(train_image_dir):
        shutil.rmtree(train_image_dir)
    os.makedirs(train_image_dir)
    test_image_dir = f".\\{yolo_path}\\images\\test"
    if os.path.exists(test_image_dir):
        shutil.rmtree(test_image_dir)
    os.makedirs(test_image_dir)
    train_label_dir = f".\\{yolo_path}\\labels\\train"
    if os.path.exists(train_label_dir):
        shutil.rmtree(train_label_dir)
    os.makedirs(train_label_dir)
    test_label_dir = f".\\{yolo_path}\\labels\\test"
    if os.path.exists(test_label_dir):
        shutil.rmtree(test_label_dir)
    os.makedirs(test_label_dir)


    with open('./annotations/train.json', 'r') as f:
        train_json_data = json.load(f)
    with open('./annotations/test.json', 'r') as f:
        test_json_data = json.load(f)

    print("train data len:", len(train_json_data['images']))
    print("test data len:", len(test_json_data['images']))

    cate_dict = {}
    for i, cate in enumerate(test_json_data['categories']):
        cate_dict[cate['id']] = i

    for image in test_json_data['images']:
        image_name = image['file_name']
        image_path = os.path.join(images_path, image_name)
        label_path = os.path.join(test_label_dir, image_name.replace("jpg", "txt"))

        shutil.copy(image_path, test_image_dir)
        
        image_anno = ""
        for anno in test_json_data['annotations']:
            if anno['image_id'] == image['id']:
                x_min, y_min, width, height = anno['bbox']

                x_center = x_min + width / 2
                y_center = y_min + height / 2
                
                x_center_norm = round(x_center / image['width'], 6)
                y_center_norm = round(y_center / image['height'], 6)
                
                width_norm = round(width / image['width'], 6)
                height_norm = round(height / image['height'], 6)

                text = f"{cate_dict[anno['category_id']]} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"

                image_anno = image_anno + text

        with open(label_path, 'w') as file:
            file.write(image_anno)

    for image in train_json_data['images']:
        image_name = image['file_name']
        image_path = os.path.join(images_path, image_name)
        label_path = os.path.join(train_label_dir, image_name.replace("jpg", "txt"))

        shutil.copy(image_path, train_image_dir)
        
        image_anno = ""
        for anno in train_json_data['annotations']:
            if anno['image_id'] == image['id']:
                x_min, y_min, width, height = anno['bbox']

                x_center = x_min + width / 2
                y_center = y_min + height / 2
                
                x_center_norm = round(x_center / image['width'], 6)
                y_center_norm = round(y_center / image['height'], 6)
                
                width_norm = round(width / image['width'], 6)
                height_norm = round(height / image['height'], 6)

                text = f"{cate_dict[anno['category_id']]} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"

                image_anno = image_anno + text

        with open(label_path, 'w') as file:
            file.write(image_anno)


def make_yaml(data_path, yaml_name, category_data):
    file_path = f'.\\{data_path}\\{yaml_name}.yaml'

    path_text = f'path: ../{data_path}\n'
    train_text = 'train: images/train\n'
    val_text = 'val: images/test\n'
    test_text = 'test: \n\n'

    class_text = 'names: \n'
    for i, category in enumerate(category_data):
        class_text = class_text + f'    {i}: {category["name"]}\n'
    
    write_text = path_text + train_text + val_text + test_text + class_text

    with open(file_path, 'w') as file:
        file.write(write_text)


def step():
    # 이미지셋 디렉토리 구조 변경
    change_image_structure(images_path='.\\mydata', move_path='.\\mydata\\images')

    # labelme 포멧 coco 포멧으로 변경
    labelme2coco(data_dir=".\\mydata\\images")

    # coco데이터셋 저장
    with open('labelme2coco.json', 'r') as f:
        json_data = json.load(f)

    # 저장된 이미지 디렉토리 설정
    images_path = ".\\mydata\\images"

    # 데이터셋 필터링 및 분리
    cleaning_dataset(json_data=json_data)
    split_dataset(images_path=images_path, json_data=json_data)

    # 데이터셋 yolo 구조로 변경
    manual_convert(images_path=".\\mydata\\images", yolo_path="pp")
    make_yaml(data_path="pp", yaml_name="pp_convert", category_data=json_data['categories'])

    print("Start yolo training command: yolo detect train data=pp_convert.yaml model=yolov8n.pt epochs=200 imgsz=640")


if __name__ == "__main__":
    step()