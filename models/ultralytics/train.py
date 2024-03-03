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

    config = {  "architecture" : f"{cfg['model']}-{cfg['weight'].split('.pt')[0]}",
                "config"       : cfg['config_file'],
                "dataset"      : cfg['train'].split('/')[0] ,
                "epochs"       : cfg['epochs'],
                "pretrain"     : cfg['weight'],
                "num_workers"  : cfg['num_workers'],
                "batch_size"   : cfg['batch_size'],
                "num_classes"  : len(cfg['names']), 
                "output_dir"   : cfg['save_dir'], 
            }
    shutil.copy(cfg['config_file'], cfg['save_dir'])

    if cfg['wandb']:
        wandb.init( project=cfg['project'],
                    name = cfg['run_name'],
                    config=config )

    model_selection = {
        "yolo": YOLO(cfg['weight']),
        "rtdetr": RTDETR(cfg['weight'])
    }

    model = model_selection.get(cfg['model'], None)
    if model is None:
        print("Invalid model type")

    
    start_time = time.time()
    model.train( wandb if cfg['wandb'] else None,
                 data = cfg['config_file'], 
                 device = cfg['device'],
                 epochs = cfg['epochs'], 
                 batch = cfg['batch_size'], 
                 workers = cfg['num_workers'],
                 optimizer = config['optimizer'],
                 lr0 = config['lr0'],
                 project = cfg['project'],
                 name = cfg['run_name'],
            )  
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time : {}'.format(total_time_str))
    config["training_time"] = total_time_str

    # with open(os.path.join(cfg['save_dir'],'training_config.yaml'), 'w') as file:
    #     yaml.dump(config, file)
    # print("Outputs and results saved to ", cfg['save_dir'])
    
    if cfg['wandb']:
        wandb.finish() 