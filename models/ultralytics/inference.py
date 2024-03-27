import os
import numpy as np
from ultralytics import YOLO, RTDETR

def inference(**cfg):

    if os.path.exists(cfg['inference']['weight']):

        model_selection = { "yolov8": YOLO(cfg['inference']['weight']),
                            "rtdetr": RTDETR(cfg['inference']['weight'])
                          }

        model = model_selection.get(cfg['model'], None)
        if model is None:
            print("Invalid model type")
        print(cfg['project'], cfg['run_name'])
        predictions = model.predict(source = cfg['inference']['source'], 
                                    stream = False, 
                                    device = cfg['device'], 
                                    save=cfg['inference']['save'], 
                                    conf = cfg['inference']['conf'],
                                    iou = cfg['inference']['iou'],
                                    max_det = cfg['inference']['max_det'], 
                                    show = cfg['inference']['show'], 
                                    imgsz = cfg['inference']['imgsz'],
                                    half = cfg['inference']['half'],
                                    visualize = cfg['inference']['visualize'],
                                    project = cfg['project'],
                                    name = cfg['run_name'],
                                    verbose =True)
    else:
        print("Model weights file not found. Please refer to the README for downloading the weight file.")