import os
import sys
import yaml
import argparse
import importlib
from types import SimpleNamespace

from utils.files import get_save_dir

sys.path.append(os.path.abspath('models/ultralytics'))
from models.ultralytics import train as yolo_train, val as yolo_eval

sys.path.append(os.path.abspath('models/efficientdet'))
from models.efficientdet import train as efficientdet_train #, val as yolo_eval

def get_arguments():
    parser = argparse.ArgumentParser(description='Script to run different models with specified modes.')
    parser.add_argument('--model', choices=['yolo', 'efficient-det', 'detr', 'rt-detr'], required=True,
                        help='Choose the model to use from the list: yolov8, efficient-det, detr, rt-detr.')
    parser.add_argument('--mode', choices=['train', 'val', 'inference'], required=True,
                        help='Specify the mode to run the model: train, val, or inference.')
    parser.add_argument('--source', default='data',
                        help='Input source for inference mode, such as image or video path. only require if mode is inference')
    return parser.parse_args()

# Function to load configuration from a YAML file
def load_config(model_name, mode_name):
    config_path = os.path.join('configs', f'{model_name}.yml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    return {}

# Mapping of models to their train and eval functions
model_functions = {
    'yolo': {'train': yolo_train.train,  'val': yolo_eval.val },
    'efficient-det' : {'train': efficientdet_train.train},
#    'detr': {'train': detr_train.main, 'eval': detr_eval.main},
}

model_configs = {
    'yolo'          : 'configs/yolo.yml',   
    'efficient-det' : 'configs/efficient-det.yml',
}

def run_task(args ,config ):

    #config['project'] = f"runs/{config['project']}"
    if args.mode == "inference":
        save_dir = get_save_dir(config['project'], "outputs")
    else:
        save_dir = get_save_dir(config['project'], config['run_name'])
    print(f"Results will be saved to {save_dir}")

    task_config = { **config, **vars(args), 
                    'config_file' : model_configs[args.model], 
                    'save_dir' : save_dir} 

    config = SimpleNamespace(**task_config)
    task_function = model_functions[config.model][config.mode]  # Get the function from the mapping
    task_function(**task_config)


if __name__ == "__main__":

    args = get_arguments()
    config = load_config(args.model, args.mode)

    run_task(args, config)