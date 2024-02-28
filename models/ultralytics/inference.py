import cv2
import argparse
import numpy as np
import os
import time
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.files import increment_path

# def get_args():
#     parser = argparse.ArgumentParser(description='')
#     #parser.add_argument('--model', type=str, default="yolo", help='training model, yolo or rtdetr')
#     parser.add_argument('--config', type=str, default="configs/adult-kid.yaml", help='path to the training config file')
#     parser.add_argument('--weight', default='weights/best.pt',type=str, help='pretrained model')
#     parser.add_argument('--source', type=str, required=True, help='path to images and videos dir')
#     parser.add_argument('--save_txt', action='store_true', help='Save predictions into a txt file.')
#     parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold for detection')
#     parser.add_argument('--nms_thres', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
#    return parser.parse_args()

def inference(cfg):

    args = get_args()
    save_dir = increment_path(Path('runs/detect/predict'), exist_ok=False, sep='', mkdir=False)
    if os.path.exists(args.weight):

        model = YOLO(model=args.weight)

        predictions = model.predict(source=args.source, 
                                    stream=True,  
                                    save=False, 
                                    conf = args.conf_thres,
                                    iou = args.nms_thres,
                                    max_det=100, 
                                    show=False, 
                                    device=0,
                                    imgsz=[640, 640],
                                    verbose =True)
        for result in predictions:
            if args.save_txt:
                txt_path = os.path.join(save_dir, 'txts', result.path.split('/')[-1].replace('.jpg', '.txt'))
                result.save_txt(txt_path)
    else:
        print("Model weights file not found. Please refer to the README for downloading the weight file.")