import os
import json
import glob

def step():
    data_dir = ".\\images"
    data = glob.glob(os.path.join(data_dir + '/*/*.json'))
    print("data len:", len(data))

    make_coco = {'info': {}, 'licenses': [], 'images': [], 'annotations': [], 'categories': []}

    for img_id in range(len(data)):

        with open(data[img_id], 'r') as f:
            load_json = json.load(f)

        make_coco['images'].append({'license': 0, 'file_name': data[img_id].replace('json','jpg'), 'coco_url': "", 'height': load_json['imageHeight'], 'width': load_json['imageWidth'], 'data_captured': 0, 'flickr_url': "", "id": img_id})

        for box_id, box in enumerate(load_json['shapes']):
            category_id_list = [category['name'] for category in make_coco['categories']]
            if box['label'] not in category_id_list:
                make_coco['categories'].append({'supercategory': 0, 'id': len(category_id_list), 'name': box['label']})
                category_id_list = [category['name'] for category in make_coco['categories']]

            x1, y1 = min(box['points'][0][0], box['points'][1][0]), min(box['points'][0][1], box['points'][1][1])
            x2, y2 = max(box['points'][0][0], box['points'][1][0]), max(box['points'][0][1], box['points'][1][1])
            w, h = x2 - x1, y2 - y1
            make_coco['annotations'].append({'segmentation': [], 'area': w * h, 'iscrowd': 0, 'image_id': img_id, 'bbox': [x1, y1, w, h], 'category_id': category_id_list.index(box['label']), 'id': box_id})

    with open('labelme2coco.json', 'w') as make_file:
        json.dump(make_coco, make_file)

if __name__ == "__main__":
    step()