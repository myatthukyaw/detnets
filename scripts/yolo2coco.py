import os
import json
import argparse
from PIL import Image
from tqdm import tqdm


def yolo_to_coco(image_dir, label_dir, output_dir, split_set=None):
    # Define categories
    categories = [{'id': 1, 'name': 'adult'}, {'id': 2, 'name': 'kid'}]


    # Initialize data dict
    data = {'train': [], 'validation': []}  # 'test': []}

    # Loop over splits
    for split in [split_set]: 
        split_data = {'info': {}, 'licenses': [], 'images': [], 'annotations': [], 'categories': categories}

        # Get image and label files for current split
        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))

        # Loop over images in current split
        cumulative_id = 0
        with tqdm(total=len(image_files), desc=f'Processing {split} images') as pbar:
            for i, filename in enumerate(image_files):
                image_path = os.path.join(image_dir, filename)
                im = Image.open(image_path)
                im_id = i + 1

                split_data['images'].append({
                    'id': im_id,
                    'file_name': filename,
                    'width': im.size[0],
                    'height': im.size[1]
                })

                # Get labels for current image
                label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
                if not os.path.exists(label_path):
                    continue
                with open(label_path, 'r') as f:
                    yolo_data = f.readlines()

                for line in yolo_data:
                    class_id, x_center, y_center, width, height = line.split()
                    class_id = int(class_id) + 1 
                    bbox_x = (float(x_center) - float(width) / 2) * im.size[0]
                    bbox_y = (float(y_center) - float(height) / 2) * im.size[1]
                    bbox_width = float(width) * im.size[0]
                    bbox_height = float(height) * im.size[1]

                    split_data['annotations'].append({
                        'id': cumulative_id,
                        'image_id': im_id,
                        'category_id': class_id,
                        'bbox': [bbox_x, bbox_y, bbox_width, bbox_height],
                        'area': bbox_width * bbox_height,
                        'iscrowd': 0
                    })

                    cumulative_id += 1

                pbar.update(1)

        data[split] = split_data

    for split in [split_set]:
        filename = os.path.join(output_dir, f'{split}.json')
        with open(filename, 'w') as f:
            json.dump(data[split], f)

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_root', type=str, default='datasets/adult-kid-v3.1-base/')
    parser.add_argument('--output_dataset', type=str, default='datasets/adult-kid-v3.1-base-coco-format/')
    return parser.parse_args()

if __name__ == "__main__": 

    args = get_args()

    for set in ['train', 'val']:

        image_dir = os.path.join( args.dataset_root ,'images', set)
        label_dir = os.path.join( args.dataset_root ,'labels', set)
        output_dir = os.path.join( args.outut_dataset, 'annotations')
        yolo_to_coco(image_dir, label_dir, output_dir, split_set=set)
    print("Conversion completed.")