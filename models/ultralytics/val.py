import time
import datetime
from ultralytics import YOLO, RTDETR

def val(**cfg):

    model_selection = {
        "yolo": YOLO(cfg['weight']),
        "rtdetr": RTDETR(cfg['weight'])
    }

    model = model_selection.get(cfg['model'], None)
    if model is None:
        print("Invalid model type")

    
    start_time = time.time()
    metrics = model.val(  data = cfg['config_file'], 
                          device = cfg['device'],
                          batch = cfg['val']['batch_size'], 
                          conf = cfg['inference']['conf'],
                          iou = cfg['inference']['iou'],
                          max_det = cfg['inference']['max_det'], 
                          save_json = cfg['inference']['save_json'], 
                          imgsz = cfg['inference']['imgsz'],
                          half = cfg['inference']['half'],
            )  
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Validation time : {}'.format(total_time_str))