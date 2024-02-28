import argparse
import os
import cv2

#for verify_load_data
import numpy as np 
import random
from detectron2.utils.visualizer import Visualizer

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

#for evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def setup_cfg(args):
    #load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = "output/model_final.pth" #model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml") #initialize a pretrained weights
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(["adult", "kid"])
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="adultkid datasets for Detectron2")
    parser.add_argument(
        "--config-file",
        default="configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
        )
    parser.add_argument(
        "--adult_kid_dir",
        default="./",
        help="path to egohands dataset"
        )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_dicts(dataset_dir, set):
    import json
    from detectron2.structures import BoxMode

    if set == "train":
        json_file = os.path.join(dataset_dir, "annotations/instances_train.json") # Update with your JSON file name
        dataset_set_dir = os.path.join(dataset_dir,set)
    elif set=="val":
        json_file = os.path.join(dataset_dir, "annotations/instances_val.json") # Update with your JSON file name
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
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    #trainer.train()
    return trainer

def verify_load_data(args, num_samples, egohands_metadata):
    dataset_dicts = get_dicts(args.egohands_dir + "train")
    for d in random.sample(dataset_dicts, num_samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=egohands_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def evaluate(cfg, trainer):
    evaluator = COCOEvaluator("adult_kid_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "adult_kid_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    return results

def main():
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    for d in ["train", "val"]:
        DatasetCatalog.register("adult_kid_" + d, lambda d=d: get_dicts("../datasets/adult-kid-v3.1-base-coco-format/", d))
        MetadataCatalog.get("adult_kid_" + d).set(thing_classes=["adult", "kid"])
    adult_kid_metadata = MetadataCatalog.get("adult_kid_train")
    #print(egohands_metadata.things_classes)
    #verify_load_data(args, num_samples=3, egohands_metadata=egohands_metadata)

    cfg.DATASETS.TRAIN = ("adult_kid_train", )
    cfg.DATASETS.TEST = ("adult_kid_val", )
    trainer = train_net(cfg)
    #evaluate
    results = evaluate(cfg, trainer) #ONLY THIS LINE MAKE ERROR!
    return 0

if __name__ == "__main__":
    main()
