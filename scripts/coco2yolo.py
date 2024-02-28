import os
import tqdm
import argparse
from pycocotools.coco import COCO


def convert_coco_to_yolo(coco_json_path, output_dir):

    coco = COCO(coco_json_path)

    os.makedirs(output_dir, exist_ok=True)

    # Get all image ids
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        # Open a text file for writing YOLO formatted annotations
        with open(os.path.join(output_dir, f"{os.path.splitext(img_info['file_name'])[0]}.txt"), 'w') as f:
            for ann in anns:
                # Get the category id and bbox coordinates
                cat_id = ann['category_id']
                bbox = ann['bbox']  # COCO format: [top left x, top left y, width, height]

                # Convert COCO bbox format to YOLO format
                x_center = (bbox[0] + bbox[2] / 2) / img_info['width']
                y_center = (bbox[1] + bbox[3] / 2) / img_info['height']
                width = bbox[2] / img_info['width']
                height = bbox[3] / img_info['height']

                # COCO class ids start at 1, subtract 1 for 0-indexed (if necessary)
                class_id = cat_id - 1

                # Write the YOLO formatted annotation to file
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--coco_dataset_root', type=str, default='datasets/SSLAD-2D')
    parser.add_argument('--output_yolo_dataset', type=str, default='datasets/SSLAD-2D-YOLO/')
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    annotations_path = os.path.join(args.coco_dataset_root, 'annotations')
    annotations = os.listdir(annotations_path)

    for anno in annotations:
        anno_path = os.path.join(annotations_path, anno)

        yolo_output_dir = os.path.join(args.output_yolo_dataset, 'labels', anno.split('.')[0].split('_')[-1])

        convert_coco_to_yolo(anno_path, yolo_output_dir)
    print("Conversion completed.")
