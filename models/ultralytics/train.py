import os
import time
import yaml
import datetime
import shutil
import wandb
from pathlib import Path
from types import SimpleNamespace
from ultralytics import YOLO, RTDETR
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils.files import increment_path

def train(**cfg):

    config = {  "architecture" : f"{cfg['model']}-{cfg['train_cfg']['weight'].split('.pt')[0]}",
                "config"       : cfg['config_file'],
                "dataset"      : cfg['train'].split('/')[0] ,
                "epochs"       : cfg['train_cfg']['epochs'],
                "pretrain"     : cfg['train_cfg']['weight'],
                "num_workers"  : cfg['train_cfg']['workers'],
                "batch_size"   : cfg['train_cfg']['batch_size'],
                "num_classes"  : len(cfg['names']), 
                "output_dir"   : cfg['save_dir'], 
            }

    if cfg['wandb']:
        wandb.init( project=cfg['project'],
                    name = cfg['run_name'],
                    config=config )

    model_selection = {
        "yolo": YOLO(cfg['train_cfg']['weight']),
        "rt-detr": RTDETR(cfg['train_cfg']['weight'])
    }

    model = model_selection.get(cfg['model'], None)
    if model is None:
        print("Invalid model type")

    
    start_time = time.time()
    model.train( wandb if cfg['wandb'] else None,
                 data = cfg['config_file'], 
                 device = cfg['device'],
                 epochs = cfg['train_cfg']['epochs'], 
                 batch = cfg['train_cfg']['batch_size'], 
                 workers = cfg['train_cfg']['workers'],
                 optimizer = cfg['train_cfg']['optimizer'],
                 lr0 = cfg['train_cfg']['lr0'],
                 lrf = cfg['train_cfg']['lrf'],
                 patience = cfg['train_cfg']['patience'],
                 imgsz = cfg['train_cfg']['imgsz'],
                 save = cfg['train_cfg']['save'],
                 save_period = cfg['train_cfg']['save_period'],
                 cache = cfg['train_cfg']['cache'],
                 verbose = cfg['train_cfg']['verbose'],
                 seed = cfg['train_cfg']['seed'],
                 cos_lr = cfg['train_cfg']['cos_lr'],
                 close_mosaic = cfg['train_cfg']['close_mosaic'],
                 profile = cfg['train_cfg']['profile'],
                 momentum = cfg['train_cfg']['momentum'],
                 plots = cfg['train_cfg']['plots'],
                 project = cfg['project'],
                 name = cfg['run_name'],
            )  
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time : {}'.format(total_time_str))
    config["training_time"] = total_time_str
    
    shutil.copy(cfg['config_file'], cfg['save_dir'])

    # with open(os.path.join(cfg['save_dir'],'training_config.yaml'), 'w') as file:
    #     yaml.dump(config, file)
    # print("Outputs and results saved to ", cfg['save_dir'])
    
    if cfg['wandb']:
        wandb.finish() 