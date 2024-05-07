import os
import json
import glob

def step():
    data_dir = ".\\images"
    data = glob.glob(os.path.join(data_dir + '/*/*.json'))
    print("data len:", len(data))

    make_coco = {'images': [], 'annotations': [], 'categories': []}

    for img_id in range(len(data)):

        with open(data[img_id], 'r') as f:
            load_json = json.load(f)

        make_coco['images'].append({'license': 0,
                                    'file_name': data[img_id].replace('json','jpg'),
                                    'height': load_json['imageHeight'],
                                    'width': load_json['imageWidth'],
                                    "id": img_id,
                                    })

        for box_id, box in enumerate(load_json['shapes']):
            category_id_list = [category['name'] for category in make_coco['categories']]
            if box['label'] not in category_id_list:
                make_coco['categories'].append({'id': len(category_id_list), 'name': box['label']})
                category_id_list = [category['name'] for category in make_coco['categories']]

            x1, y1 = min(box['points'][0][0], box['points'][1][0]), min(box['points'][0][1], box['points'][1][1])
            x2, y2 = max(box['points'][0][0], box['points'][1][0]), max(box['points'][0][1], box['points'][1][1])
            w, h = x2 - x1, y2 - y1
            make_coco['annotations'].append({'id': box_id,
                                             'image_id': img_id,
                                             'category_id': category_id_list.index(box['label']),
                                             'area': w * h,
                                             'bbox': [x1, y1, w, h],
                                             'iscrowd': 0,
                                             })

    with open('labelme2coco.json', 'w') as make_file:
        json.dump(make_coco, make_file)

if __name__ == "__main__":
    step()