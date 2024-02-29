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
                          batch = cfg['batch_size'], 
                          project = cfg['project'],
            )  
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Validation time : {}'.format(total_time_str))