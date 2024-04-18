from ultralytics.data.converter import convert_coco
import json
import os
import glob
import random
import shutil


def cleaning_dataset(json_data: dict):
    print("Cleaning Dataset")

    for image in json_data['images']:
        bboxes = []
        for anno in json_data['annotations']:
            if image['id'] == anno['image_id']:
                bboxes.append(anno)
        if len(bboxes) == 0:
            image['license'] = -1

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
    train_json['info'] = json_data['info']
    train_json['licenses'] = json_data['licenses']
    train_json['categories'] = json_data['categories']
    train_json['images'] = []
    train_json['annotations'] = []

    test_json = {}
    test_json['info'] = json_data['info']
    test_json['licenses'] = json_data['licenses']
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

def auto_convert(images_path):
    print("Auto Convert")
    save_dir = ".\\coco_convert_auto"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    convert_coco(labels_dir='.\\annotations', save_dir=save_dir)\
    
    images = glob.glob(os.path.join(f"{images_path}\\*.jpg"))

    train_anno_dir = ".\\coco_convert_auto\\labels\\train"
    test_anno_dir = ".\\coco_convert_auto\\labels\\test"
    train_anno_texts = glob.glob(os.path.join(train_anno_dir, "*.txt"))
    test_anno_texts = glob.glob(os.path.join(test_anno_dir, "*.txt"))

    train_anno_list = []
    for anno_text in train_anno_texts:
        train_anno_list.append(os.path.basename(anno_text.replace(".txt","")))
    test_anno_list = []
    for anno_text in test_anno_texts:
        test_anno_list.append(os.path.basename(anno_text.replace(".txt","")))

    print(len(train_anno_list))
    print(len(test_anno_list))

    train_image_dir = ".\\coco_convert_auto\\images\\train"
    os.makedirs(train_image_dir)
    test_image_dir = ".\\coco_convert_auto\\images\\test"
    os.makedirs(test_image_dir)

    for image in images:
        image_name = os.path.basename(image.replace(".jpg",""))
        if image_name in test_anno_list:
            shutil.copy(image, test_image_dir)
        else:
            shutil.copy(image, train_image_dir)

    print("Convert Done")

def manual_convert(images_path):
    train_image_dir = "./coco_convert_manual/images/train"
    if os.path.exists(train_image_dir):
        shutil.rmtree(train_image_dir)
        os.makedirs(train_image_dir)
    test_image_dir = "./coco_convert_manual/images/test"
    if os.path.exists(test_image_dir):
        shutil.rmtree(test_image_dir)
        os.makedirs(test_image_dir)
    train_label_dir = "./coco_convert_manual/labels/train"
    if os.path.exists(train_label_dir):
        shutil.rmtree(train_label_dir)
        os.makedirs(train_label_dir)
    test_label_dir = "./coco_convert_manual/labels/test"
    if os.path.exists(test_label_dir):
        shutil.rmtree(test_label_dir)
        os.makedirs(test_label_dir)


    with open('./annotations/train.json', 'r') as f:
        train_json_data = json.load(f)
    with open('./annotations/test.json', 'r') as f:
        test_json_data = json.load(f)

    print(len(train_json_data['images']))
    print(len(test_json_data['images']))

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
    file_path = f'./{data_path}/{yaml_name}.yaml'

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
    with open('instances_val2017.json', 'r') as f:
        json_data = json.load(f)

    images_path = ".\\val2017"

    cleaning_dataset(json_data=json_data)
    split_dataset(images_path=images_path, json_data=json_data)

    auto_convert(images_path=images_path)
    make_yaml(data_path="coco_convert_auto", yaml_name="auto_convert", category_data=json_data['categories'])

    manual_convert(images_path=images_path)
    make_yaml(data_path="coco_convert_manual", yaml_name="manual_convert", category_data=json_data['categories'])


if __name__ == "__main__":
    step()