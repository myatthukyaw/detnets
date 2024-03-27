import json
import os

import argparse
import torch
import yaml
from tqdm import tqdm
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from utils.utils import preprocess, invert_affine, postprocess, boolean_string


class Eval:
    def __init__(self, **cfg):
        self.cfg = cfg
        self.compound_coef = cfg['compound_coef']
        self.anchor_ratios = eval(self.cfg['anchors_ratios'])
        self.anchor_scales = eval(self.cfg['anchors_scales'])
        self.threshold = self.cfg['test']['conf_thres']
        self.nms_threshold = self.cfg['test']['nms_thres']
        self.use_float16 = self.cfg['test']['use_float16']
        
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self._get_input_sizes()
        self._get_val_set_generator()
        self._setup_model()
        self._move_to_device()
    
    def _setup_model(self):
        self.model = EfficientDetBackbone(num_classes=len(self.cfg['obj_list']), 
                                          compound_coef=self.cfg['compound_coef'],
                                          ratios=eval(self.cfg['anchors_ratios']), 
                                          scales=eval(self.cfg['anchors_scales']))
        self.model.load_state_dict(torch.load(self.cfg['test']['weight'], map_location=torch.device('cpu')))
        self.model.requires_grad_(False)
        self.model.eval()

    def _move_to_device(self):
        # moving model to device
        self.model = self.model.to(self.cfg['test']['device'])
        if self.use_float16:
            self.model.half()


    def _get_input_sizes(self):
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    
    def _create_dataset(self, set_name):
        dataset_transforms = transforms.Compose([
            Normalizer(mean=self.cfg['mean'], std=self.cfg['std']),
            Augmenter(),
            Resizer(self.input_sizes[self.cfg['compound_coef']])
        ])

        return CocoDataset(
            root_dir=os.path.join(self.cfg['train']['data_path'], self.cfg['project_name']), 
            set=set_name,
            transform=dataset_transforms
        )

    def _get_dataloader_params(self, shuffle):
        return {
            'batch_size': self.cfg['test']['batch_size'],
            'shuffle': shuffle,
            'drop_last': True,
            'collate_fn': collater,
            'num_workers': self.cfg['test']['num_workers']
        }
        
    def _get_val_set_generator(self):
        val_params = self._get_dataloader_params(shuffle=False)
        val_set = self._create_dataset(set_name=self.cfg['val_set'])
        self.val_generator = DataLoader(val_set, **val_params)

    def _aspectaware_resize_padding(self, image_shape, width, height):
        old_h, old_w, c = image_shape

        if old_w > old_h:
            new_w = width
            new_h = int(width / old_w * old_h)
        else:
            new_w = int(height / old_h * old_w)
            new_h = height

        return new_w, new_h, old_w, old_h, 0, 280

    def run_cocoeval(self, pred_json_path):

        VAL_GT = os.path.join(self.cfg['train']['data_path'], self.cfg['project_name'], 'annotations/instance_val.json')
        coco_gt = COCO(VAL_GT)
        image_ids = coco_gt.getImgIds()
        # load results in COCO evaluation tool
        coco_pred = coco_gt.loadRes(pred_json_path)

        # run COCO evaluation
        print('BBox')
        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval
    
    def _write_eval_results(self, imgs, imgs_meta, imgs_ids, res):
        """
            Write evaluation results to a JSON file.
            Args:
                imgs (list): List of input images.
                imgs_meta (list): List of image metadata.
                imgs_ids (list): List of image IDs.
                res (tuple): Tuple containing regression, classification, and anchor outputs.
            Returns:
                None
        """
        regressions, classifications, anchors = res

        for img, reg, cls, img_meta, img_id in zip(imgs, regressions, classifications, imgs_meta, imgs_ids):
            preds = self._postprocess_predictions(img, reg, cls, anchors, img_meta)[0]
            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']

            if rois.shape[0] > 0:
                # x1,y1,x2,y2 -> x1,y1,w,h
                rois[:, 2] -= rois[:, 0]
                rois[:, 3] -= rois[:, 1]

                bbox_score = scores
                for roi_id in range(rois.shape[0]):
                    score = float(bbox_score[roi_id])
                    label = int(class_ids[roi_id])
                    box = rois[roi_id, :]

                    image_result = {
                        'image_id': img_id,
                        'category_id': label + 1,
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    self.results.append(image_result)

    def _postprocess_predictions(self, img, reg, cls, anchors, img_meta):
        preds = postprocess(img.unsqueeze(0), anchors, reg.unsqueeze(0), cls.unsqueeze(0),
                            self.regressBoxes, self.clipBoxes, self.threshold, self.nms_threshold)
        preds = invert_affine([img_meta], preds)
        return preds

    def run_val(self):
        self.model.eval()
        self.results = []
        progress_bar = tqdm(self.val_generator)

        for iter, data in enumerate(progress_bar):
            imgs, annot, imgs_ids, imgs_meta = self._prepare_data(data)
            res = self._infer_and_process(imgs, annot)
            self._write_eval_results(imgs, imgs_meta, imgs_ids, res)

        pred_json_path = self._save_results_to_json()
        print("Prediction json file saved to ", pred_json_path)
        self.run_cocoeval(pred_json_path)
        #return pred_json_path

    def _prepare_data(self, data):
        imgs = data['img']
        annot = data['annot']
        imgs_ids = data['img_id']

        imgs = imgs.to(self.cfg['test']['device'])
        annot = annot.to(self.cfg['test']['device'])

        max_size = self.input_sizes[self.cfg['compound_coef']]
        imgs_meta = [self._aspectaware_resize_padding(img_shape, max_size, max_size) for img_shape in data['orig_shape']]

        return imgs, annot, imgs_ids, imgs_meta

    def _infer_and_process(self, imgs, annot):
        with torch.no_grad():
            features, regression, classification, anchors = self.model(imgs)
        return [regression, classification, anchors]

    def _save_results_to_json(self):
        pred_json_path = os.path.join(self.cfg['train']['save_dir'], 'val_bbox_results.json')
        if os.path.exists(pred_json_path):
            os.remove(pred_json_path)
        json.dump(self.results, open(pred_json_path, 'w'), indent=4)
        return pred_json_path


def test(**cfg):

    evaluation = Eval(**cfg)
    evaluation.run_val()