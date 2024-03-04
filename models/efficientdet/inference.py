import ast
import time

import cv2
import torch
import numpy as np
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

class Inference:
    def __init__(self, **cfg):
        self.cfg = cfg
        self.compound_coef = cfg['compound_coef']
        self.source = cfg['source']
        self.anchor_ratios = eval(self.cfg['anchors_ratios'])
        self.anchor_scales = eval(self.cfg['anchors_scales'])
        self.threshold = self.cfg['conf_thres']
        self.iou_threshold = self.cfg['iou_thres']
        self.use_cuda = True if cfg['num_gpus'] > 0 else False
        self.use_float16 = self.cfg['use_float16']
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        cudnn.fastest = True
        cudnn.benchmark = True

        self.obj_list = self.cfg['obj_list']
        self.color_list = standard_to_bgr(STANDARD_COLORS)
        self._load_model()
        self.input_sizes = self._get_input_sizes()
        self.input_size = self.input_sizes[self.compound_coef] # if force_input_size is None else force_input_size

    def _get_input_sizes(self):
        return [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    
    def _load_model(self):
        self.model = EfficientDetBackbone( compound_coef=self.compound_coef, 
                                           num_classes=len(self.obj_list),
                                           ratios=self.anchor_ratios, 
                                           scales=self.anchor_scales)
        self.model.load_state_dict( torch.load(self.cfg['inf_weight'], 
                                    map_location='cpu'))
        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            model = model.cuda()
        if self.use_float16:
            model = model.half()
    
    def run_inference(self):

        ori_imgs, framed_imgs, framed_metas = preprocess(self.source, max_size=self.input_size)
        
        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)
            out = postprocess( x, anchors, regression, classification, self.regressBoxes, 
                               self.clipBoxes, self.threshold, self.iou_threshold)

        out = invert_affine(framed_metas, out)
        self.display(out, ori_imgs, imshow=self.cfg['show'], imwrite=self.cfg['save'])


    def display(self, preds, imgs,  imshow=True, imwrite=False):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue

            imgs[i] = imgs[i].copy()

            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int64)
                obj = self.obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                plot_one_box( imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                              color=self.color_list[get_index_label(obj, self.obj_list)])

            if imshow:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)

            if imwrite:
                cv2.imwrite(f'test/img_inferred_d{self.compound_coef}_this_repo_{i}.jpg', imgs[i])



def inference(**cfg):
    
    infer = Inference(**cfg)
    infer.run_inference()