import os
import json
import argparse
from PIL import Image
from tqdm import tqdm


def yolo_to_coco(image_dir, label_dir, output_path, categories):
    """Convert a single YOLO split (images + label txt files) to one COCO JSON file."""
    coco = {'info': {}, 'licenses': [], 'images': [], 'annotations': [], 'categories': categories}

    image_files = sorted(os.listdir(image_dir))

    annotation_id = 0
    for i, filename in enumerate(tqdm(image_files, desc=f'Processing {os.path.basename(image_dir)}')):
        image_path = os.path.join(image_dir, filename)
        with Image.open(image_path) as im:
            width, height = im.size
        image_id = i + 1

        coco['images'].append({
            'id': image_id,
            'file_name': filename,
            'width': width,
            'height': height,
        })

        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            yolo_data = f.readlines()

        for line in yolo_data:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, w, h = parts
            # YOLO class ids are 0-indexed; COCO category ids here start at 1.
            category_id = int(class_id) + 1
            bbox_x = (float(x_center) - float(w) / 2) * width
            bbox_y = (float(y_center) - float(h) / 2) * height
            bbox_width = float(w) * width
            bbox_height = float(h) * height

            coco['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [bbox_x, bbox_y, bbox_width, bbox_height],
                'area': bbox_width * bbox_height,
                'iscrowd': 0,
            })
            annotation_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f)


def get_args():
    parser = argparse.ArgumentParser(description='Convert a YOLO-format dataset to COCO format.')
    parser.add_argument('--dataset_root', type=str, default='datasets/my_dataset_yolo',
                        help='Root of the YOLO dataset, containing images/<split> and labels/<split>.')
    parser.add_argument('--output_dataset', type=str, default='datasets/my_dataset_coco',
                        help='Output dataset root; annotations are written to <output_dataset>/annotations.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='Dataset splits to convert.')
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # Define COCO categories. Edit this list to match your dataset.
    categories = [{'id': 1, 'name': 'adult'}, {'id': 2, 'name': 'kid'}]

    for split in args.splits:
        image_dir = os.path.join(args.dataset_root, 'images', split)
        label_dir = os.path.join(args.dataset_root, 'labels', split)
        output_path = os.path.join(args.output_dataset, 'annotations', f'{split}.json')
        yolo_to_coco(image_dir, label_dir, output_path, categories)
    print("Conversion completed.")
