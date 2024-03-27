import argparse
import os
import cv2
import json
import yaml

import wandb
import numpy as np 
import random

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer
from detectron2.engine import HookBase

#for evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer

from modules.utils.utils import increment_path


CLASSES = ['adult', 'kid']

class WandBWriter(HookBase):
    def __init__(self):
        super().__init__()    

    def after_step(self):
        #keys = self.trainer.storage.latest().keys()
        total_loss = self.trainer.storage.latest().get('total_loss', {})
        loss_cls = self.trainer.storage.latest().get('loss_cls', {})
        loss_box_reg = self.trainer.storage.latest().get('loss_box_reg', {})
        lr = self.trainer.storage.latest().get('lr', {})
        metrics = {'total_loss': total_loss[0],
                    'loss_cls' : loss_cls[0],
                    'loss_box_reg' : loss_box_reg[0],
                    'lr' : lr[0],}
        wandb.log(metrics)

class CustomTrainer(DefaultTrainer):

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(WandBWriter())  # Add WandBWriter as a custom hook
        return hooks
    
def get_dicts(dataset_dir, set):

    json_file = os.path.join(dataset_dir, f"annotations/instances_{set}.json") # Update with your JSON file name
    dataset_set_dir = os.path.join(dataset_dir,set)
    
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns['images']):
        record = {}
        
        filename = os.path.join(dataset_set_dir, v["file_name"])
        height, width = v["height"], v["width"]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = imgs_anns['annotations']
        objs = []
        for _, anno in enumerate(annos):
            if anno['image_id'] == v['id']:
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": anno["category_id"]
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def train_net(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    metrics = trainer.storage.latest().get('metrics', {})

    return trainer

def verify_load_data(args, num_samples, adult_kid_metadata):
    dataset_dicts = get_dicts(args.dataset_dir + "train")
    for d in random.sample(dataset_dicts, num_samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=adult_kid_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def evaluate(cfg, trainer):
    evaluator = COCOEvaluator("adult_kid_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "adult_kid_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    return results


def setup_cfg(args, output_path):
    # Configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config))
    cfg.DATASETS.TRAIN = ("adult_kid_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size # This is the real "batch size"
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = args.iterations  # Adjust according to your dataset size # 10k is best for pretraining coco
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # The "RoIHead batch size". 128 is faster, and good enough for small dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(["adult", "kid"])  # Number of classes
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = str(output_path)
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="adultkid datasets for Detectron2")
    parser.add_argument("--config", default="COCO-Detection/retinanet_R_50_FPN_1x.yaml", help="config file")
    parser.add_argument("--dataset_dir", default="../../../../datasets/adult-kid-v3.1-base-coco-format/", help="path to adult kid dataset")
    parser.add_argument( "--pretrain", default="", help="path to pretrained weight file")
    parser.add_argument( "--batch_size", type=int,default=7, help="training batch size")
    parser.add_argument( "--num_workers", type=int, default=12, help="Number of workers")
    parser.add_argument( "--iterations", type=int, default=10000, help="training iterations")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER,)
    parser.add_argument("--output_dir", type=str, help="training output path", default="exps/")
    parser.add_argument("--wandb_project", default="detection-baselines" ,help="training output path")
    parser.add_argument("--wandb_run_name", default="test_run" ,help="training output path")
    return parser


def main():
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    output_path = increment_path(args.output_dir, sep='exp', mkdir=True)
    cfg = setup_cfg(args, output_path)

    config = {
        "architecture": "Retinanet-Baseline",
        "config" : args.config,
        "dataset": args.dataset_dir,
        "iterations": args.iterations,
        "pretrain" : cfg.MODEL.WEIGHTS,
        "num_workers" : args.num_workers,
        "batch_size" : args.batch_size,
        "ROI_head_batch" : cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, 
        "num_classes" : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        "base_lr" : cfg.SOLVER.BASE_LR,
        "input_format" : cfg.INPUT.FORMAT,
        "output_dir" : cfg.OUTPUT_DIR
        }

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project = args.wandb_project,
        name = args.wandb_run_name,
        # track hyperparameters and run metadata
        config=config
    )

    for d in ["train", "val"]:
        DatasetCatalog.register("adult_kid_" + d, lambda d=d: get_dicts(args.dataset_dir, d ))
        MetadataCatalog.get("adult_kid_" + d).set(thing_classes=CLASSES)
    adult_kid_metadata = MetadataCatalog.get("adult_kid_train")
    #verify_load_data(args, num_samples=3, adult_kid_metadata)

    cfg.DATASETS.TRAIN = ("adult_kid_train", )
    cfg.DATASETS.TEST = ("adult_kid_val", )

    trainer = train_net(cfg)
    results = evaluate(cfg, trainer)

    # CONFIG Write to a YAML file
    with open(os.path.join(cfg.OUTPUT_DIR,'training_config.yaml'), 'w') as file:
        yaml.dump(config, file)
    print("Outputs and results saved to ", cfg.OUTPUT_DIR)
    wandb.finish()

if __name__ == "__main__":
    main()
