import os
import ast
import time

import cv2
import torch
import numpy as np
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess,  preprocess_video, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"  # video suffixes

class Inference:
    def __init__(self, **cfg):
        self.cfg = cfg
        self.compound_coef = cfg['compound_coef']
        self.anchor_ratios = eval(self.cfg['anchors_ratios'])
        self.anchor_scales = eval(self.cfg['anchors_scales'])
        self.source = cfg['inference']['source']
        self.threshold = self.cfg['inference']['conf_thres']
        self.nms_threshold = self.cfg['inference']['nms_thres']
        self.use_cuda = True if cfg['num_gpus'] > 0 else False
        self.use_float16 = self.cfg['inference']['use_float16']
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        cudnn.fastest = True
        cudnn.benchmark = True

        self.obj_list = self.cfg['obj_list']
        self.color_list = standard_to_bgr(STANDARD_COLORS)
        self._load_model()
        self._create_save_dir()
        self.input_sizes = self._get_input_sizes()
        self.input_size = self.input_sizes[self.compound_coef] # if force_input_size is None else force_input_size

    def _get_input_sizes(self):
        return [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    
    def _create_save_dir(self):
        os.makedirs(self.cfg['save_dir'])

    def _load_model(self):
        self.model = EfficientDetBackbone( compound_coef=self.compound_coef, 
                                           num_classes=len(self.obj_list),
                                           ratios=self.anchor_ratios, 
                                           scales=self.anchor_scales)
        self.model.load_state_dict( torch.load(self.cfg['inference']['weight'], 
                                    map_location='cpu'))
        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            self.model = self.model.cuda()
        if self.use_float16:
            self.model = self.model.half()

    def _convert_to_tensor(self, framed_imgs):
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        tensors = [torch.from_numpy(img).to(device) for img in framed_imgs]
        tensor = torch.stack(tensors).to(torch.float32 if not self.use_float16 else torch.float16)
        return tensor.permute(0, 3, 1, 2)
    
    def _run_inference(self, framed_imgs, framed_metas):
        # if self.use_cuda:
        #     x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        # else:
        #     x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        # x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)
        start = time.time()
        x = self._convert_to_tensor(framed_imgs)
        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)
            out = postprocess( x, anchors, regression, classification, self.regressBoxes, 
                               self.clipBoxes, self.threshold, self.nms_threshold)
        out = invert_affine(framed_metas, out)
        end = time.time() - start
        print(f"Inference time on frame : {end}")
        return out


    def image_inference(self):
        ori_imgs, framed_imgs, framed_metas = preprocess(self.source, max_size=self.input_size)
        out = self._run_inference(framed_imgs, framed_metas)
        self.display(out, ori_imgs)
    
    def video_inference(self):
        cap = cv2.VideoCapture(self.cfg['inference']['source'])
        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=self.input_size)
            out = self._run_inference(framed_imgs, framed_metas)
            img_show = self.display(out, ori_imgs, frame_no)

            frame_no += 1 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        cap.release()
        cv2.destroyAllWindows()


    def display(self, preds, imgs, frame_no=0):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue
            #imgs[i] = imgs[i].copy()

            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int64)
                obj = self.obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                plot_one_box( imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                              color=self.color_list[get_index_label(obj, self.obj_list)])

            if self.cfg['inference']['show']:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)

            if self.cfg['inference']['save']:
                output_path = os.path.join(self.cfg['save_dir'], self.cfg['inference']['source'].split('/')[-1].split('.')[0]+f'-{frame_no}.jpg')
                cv2.imwrite(output_path, imgs[i])



def inference(**cfg):
    
    infer = Inference(**cfg)
    if cfg['inference']['source'].split(".")[-1].lower() in IMG_FORMATS:
        infer.image_inference()
    elif cfg['inference']['source'].split(".")[-1].lower() in VID_FORMATS:
        infer.video_inference()
    else:
        print("Unsupported format or dictionary.")
    print(f"Inference outputs saved to {cfg['save_dir']}")