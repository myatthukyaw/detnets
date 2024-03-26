import os
import sys
import yaml
import argparse
import importlib
from types import SimpleNamespace

from tools.files import get_save_dir

def import_model_functions(model_name):
    if model_name == 'yolov8':
        sys.path.append(os.path.abspath('models/ultralytics'))
        from models.ultralytics import train as yolov8_train, val as yolov8_eval, inference as yolov8_inference
        return {'train': yolov8_train, 'val': yolov8_eval, 'inference': yolov8_inference}

    elif model_name == 'yolov7':
        sys.path.append(os.path.abspath('models/yolov7'))
        from models.yolov7 import train as yolov7_train, val as yolov7_eval, detect as yolov7_inference
        return {'train': yolov7_train, 'val': yolov7_eval, 'inference': yolov7_inference}

    elif model_name == 'efficient-det':
        sys.path.append(os.path.abspath('models/efficientdet'))
        from models.efficientdet import train as efficientdet_train, val as efficientdet_val, inference as efficientdet_inference
        return {'train': efficientdet_train, 'val': efficientdet_val, 'inference': efficientdet_inference}

    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
def get_arguments():
    parser = argparse.ArgumentParser(description='Script to run different models with specified modes.')
    parser.add_argument('--model', choices=['yolov8', 'efficient-det', 'detr', 'rt-detr'], required=True,
                        help='Choose the model to use from the list: yolov8, efficient-det, detr, rt-detr.')
    parser.add_argument('--mode', choices=['train', 'val', 'inference'], required=True,
                        help='Specify the mode to run the model: train, val, or inference.')
    #parser.add_argument('--source', default='data',
    #                    help='Input source for inference mode, such as image or video path. only require if mode is inference')
    return parser.parse_args()

# Function to load configuration from a YAML file
def load_config(model_name):
    config_path = os.path.join('configs', f'{model_name}.yml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    return {}

model_configs = {
    'yolov8'          : 'configs/yolov8.yml',   
    'yolov7'          : 'configs/yolov7.yml',   
    'rt-detr'       : 'configs/rt-detr.yml',
    'efficient-det' : 'configs/efficient-det.yml',
}

def run_task(args ,config ):

    model_functions = import_model_functions(args.model)

    if args.mode not in model_functions:
        raise ValueError(f"Unsupported mode: {args.mode} for model: {args.model}")
    
    if args.mode == "inference":
        # inference results will be save in format - project-x/model/outputs-x
        save_dir = get_save_dir(config['project'], args.model, "outputs", sep="-")
        config['run_name'] = "outputs"
    else:
        # training results will be save in format - project-x/model/run-expx
        save_dir = get_save_dir(config['project'], args.model, config['run_name'], sep="-exp")
    
    print(f"Results will be saved to {save_dir}")

    task_config = { **config, **vars(args), 
                    'config_file' : model_configs[args.model], 
                    'save_dir' : save_dir} 

    config = SimpleNamespace(**task_config)
    task_function = model_functions[config.mode]  # Get the function from the mapping
    print(task_function)
    print(task_config)
    task_function(**task_config)


if __name__ == "__main__":

    args = get_arguments()
    config = load_config(args.model)

    run_task(args, config)