import os
import numpy as np
from ultralytics import YOLO, RTDETR

def inference(**cfg):

    if os.path.exists(cfg['weight']):

        model_selection = {
                            "yolo": YOLO(cfg['inf_weight']),
                            "rtdetr": RTDETR(cfg['inf_weight'])
                        }

        model = model_selection.get(cfg['model'], None)
        if model is None:
            print("Invalid model type")

        predictions = model.predict(source=cfg['source'], 
                                    stream=False,  
                                    save=cfg['save'], 
                                    conf = cfg['conf_thres'],
                                    iou = cfg['nms_thres'],
                                    max_det=100, 
                                    show=cfg['show'], 
                                    device=0,
                                    imgsz=[640, 640],
                                    project = cfg['project'],
                                    name = cfg['run_name'],
                                    verbose =True)
        # for result in predictions:
        #     if args.save_txt:
        #         txt_path = os.path.join(save_dir, 'txts', result.path.split('/')[-1].replace('.jpg', '.txt'))
        #         result.save_txt(txt_path)
    else:
        print("Model weights file not found. Please refer to the README for downloading the weight file.")