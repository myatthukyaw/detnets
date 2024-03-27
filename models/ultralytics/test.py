import time
import datetime
from ultralytics import YOLO, RTDETR

def test(**cfg):

    model_selection = {
        "yolov8": YOLO(cfg['test_cfg']['weight']),
        "rtdetr": RTDETR(cfg['test_cfg']['weight'])
    }

    model = model_selection.get(cfg['model'], None)
    if model is None:
        print("Invalid model type")
    
    start_time = time.time()
    metrics = model.val(  data = cfg['config_file'], 
                          device = cfg['device'],
                          batch = cfg['test_cfg']['batch_size'], 
                          conf = cfg['test_cfg']['conf'],
                          iou = cfg['test_cfg']['iou'],
                          max_det = cfg['test_cfg']['max_det'], 
                          save_json = cfg['test_cfg']['save_json'], 
                          imgsz = cfg['test_cfg']['imgsz'],
                          half = cfg['test_cfg']['half'],
            )  
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Validation time : {}'.format(total_time_str))