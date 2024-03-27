import os
import sys
import yaml
import argparse
import importlib
from types import SimpleNamespace
from tools.files import get_save_dir

def get_arguments():
    parser = argparse.ArgumentParser(description='Script to run different models with specified modes.')
    parser.add_argument('--model', choices=['yolov8', 'yolov7','efficient-det', 'detr', 'rt-detr'], required=True,
                        help='Choose the model to use from the list: yolov8, efficient-det, detr, rt-detr.')
    parser.add_argument('--mode', choices=['train', 'test', 'inference'], required=True,
                        help='Specify the mode to run the model: train, val, or inference.')
    #parser.add_argument('--source', default='data',
    #                    help='Input source for inference mode, such as image or video path. only require if mode is inference')
    return parser.parse_args()

def import_model_functions(model_name):
    if model_name == 'yolov8':
        sys.path.append(os.path.abspath('models/ultralytics'))
        from models.ultralytics import train as yolov8_train, test as yolov8_test, inference as yolov8_inference
        return { 'train': yolov8_train.train, 
                 'test': yolov8_test.test, 
                 'inference': yolov8_inference.inference
            }
    elif model_name == 'yolov7':
        sys.path.append(os.path.abspath('models/yolov7'))
        from models.yolov7 import train as yolov7_train, test as yolov7_test, detect as yolov7_inference
        return {'train': yolov7_train.train, 
                'test': yolov7_test.test, 
                'inference': yolov7_inference.detect
                }
    elif model_name == 'efficient-det':
        sys.path.append(os.path.abspath('models/efficientdet'))
        from models.efficientdet import train as efficientdet_train, test as efficientdet_test, inference as efficientdet_inference
        return {'train': efficientdet_train, 
                'test': efficientdet_test.test, 
                'inference': efficientdet_inference.inference
                }
    else:
        raise ValueError(f"Unsupported model: {model_name}")\

model_configs = {
    'yolov8'          : 'configs/yolov8.yml',   
    'yolov7'          : 'configs/yolov7.yml',   
    'rt-detr'       : 'configs/rt-detr.yml',
    'efficient-det' : 'configs/efficient-det.yml',
}

# Function to load configuration from a YAML file
def load_config(model_name):
    config_path = os.path.join('configs', f'{model_name}.yml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    return {}

# function to get save directory
def create_save_dir(args, config):
    if args.mode == "inference":
        # inference results will be save in format - project-x/model/outputs-x
        save_dir = get_save_dir(config['project'], args.model, "outputs", sep="-")
        config['project'] = f"{config['project']}/{args.model}"
        config['run_name'] = "outputs"
    else:
        # training results will be save in format - project-x/model/run-expx
        save_dir = get_save_dir(config['project'], args.model, config['run_name'], sep="-exp")
    return save_dir, config

def run_task(args ,config ):

    model_functions = import_model_functions(args.model)
    if args.mode not in model_functions:
        raise ValueError(f"Unsupported mode: {args.mode} for model: {args.model}")
    
    save_dir, config = create_save_dir(args, config)

    task_config = { **config , **vars(args), 
                    'config_file' : model_configs[args.model], 
                    'save_dir' : save_dir} 
    config = SimpleNamespace(**task_config)
    task_function = model_functions[config.mode]  # Get the function from the mapping
    task_function(**task_config)


if __name__ == "__main__":

    args = get_arguments()
    config = load_config(args.model)

    run_task(args, config)