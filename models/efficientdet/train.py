import os
import time
import json
import yaml
import shutil
import traceback
import datetime
from pathlib import Path

import wandb
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string, postprocess, invert_affine
from efficientdet.utils import BBoxTransform, ClipBoxes

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic

    if not path.exists():# and not exist_ok:
        os.mkdir(path)
    path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

    # Method 1
    for n in range(2, 9999):
        p = f'{path}/{sep}{n}{suffix}'  # increment path
        if not os.path.exists(p):  #
            break
    path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        features, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        res = (features, regression, classification, anchors)
        return cls_loss, reg_loss, res

class Train_EfficientDet:
    def __init__(self, **cfg) -> None:
        self.cfg = cfg
        self.best_loss = 1e5
        self.best_epoch = 0
        self.threshold = 0.05
        self.nms_threshold = 0.5
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        self._setup_dataset_generators()
        self._setup_model()
        self.last_step = self._load_weights()
        self.step = max(0, self.last_step)
        self._setup_optimizer()
        self._setup_scheduler()
        self._apply_sync_bn()
        self._move_to_gpu()
        self._finalize_model_setup()

    def _setup_dataset_generators(self):
        self._get_dataset_generators()
        self.num_iter_per_epoch = len(self.training_generator)

    def _setup_optimizer(self):
        self.optimizer = self._get_optimizer()

    def _setup_scheduler(self):
        self.scheduler = self._get_scheduler()

    def _setup_model(self):
        self.model = EfficientDetBackbone(num_classes=len(self.cfg['obj_list']), 
                                          compound_coef=self.cfg['compound_coef'],
                                          ratios=eval(self.cfg['anchors_ratios']), 
                                          scales=eval(self.cfg['anchors_scales']))

    def _get_optimizer(self):
        if self.cfg['optim'] == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), self.cfg['lr'])
        else:
            return torch.optim.SGD(self.model.parameters(), self.cfg['lr'], momentum=0.9, nesterov=True)

    def _get_scheduler(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

    def _apply_sync_bn(self):
        # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
        # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
        #  useful when gpu memory is limited.
        # because when bn is disable, the training will be very unstable or slow to converge,
        # apply sync_bn can solve it,
        # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
        # but it would also slow down the training by a little bit.
        if self.cfg['num_gpus'] > 1 and self.cfg['batch_size'] // self.cfg['num_gpus'] < 4:
            self.model.apply(replace_w_sync_bn)
            self.use_sync_bn = True
        else:
            self.use_sync_bn = False

    def _move_to_gpu(self):
        # moving model to GPU
        if self.cfg['num_gpus'] > 0:
            self.model = self.model.cuda()
            if self.cfg['num_gpus'] > 1:
                self.model = CustomDataParallel(self.model, self.cfg['num_gpus'])
                if self.use_sync_bn:
                    patch_replication_callback(self.model)

    def _finalize_model_setup(self):
        if self.cfg.get('head_only', False):
            self.model.apply(self._freeze_backbone)
            print('[Info] Freezed backbone')
        self.model = ModelWithLoss(self.model, debug=self.cfg.get('debug', False))

    def _freeze_backbone(self, m):
        classname = m.__class__.__name__
        for ntl in ['EfficientNet', 'BiFPN']:
            if ntl in classname:
                for param in m.parameters():
                    param.requires_grad = False

    def _get_dataset_generators(self):
        training_params = self._get_dataloader_params(shuffle=True)
        val_params = self._get_dataloader_params(shuffle=False)

        self.input_sizes = self._define_input_sizes()
        training_set = self._create_dataset(set_name=self.cfg['train_set'])
        val_set = self._create_dataset(set_name=self.cfg['val_set'])

        self.training_generator = DataLoader(training_set, **training_params)
        self.val_generator = DataLoader(val_set, **val_params)

    def _get_dataloader_params(self, shuffle):
        return {
            'batch_size': self.cfg['batch_size'],
            'shuffle': shuffle,
            'drop_last': True,
            'collate_fn': collater,
            'num_workers': self.cfg['num_workers']
        }

    def _define_input_sizes(self):
        return [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

    def _create_dataset(self, set_name):
        dataset_transforms = transforms.Compose([
            Normalizer(mean=self.cfg['mean'], std=self.cfg['std']),
            Augmenter(),
            Resizer(self.input_sizes[self.cfg['compound_coef']])
        ])

        return CocoDataset(
            root_dir=os.path.join(self.cfg['data_path'], self.cfg['project_name']), 
            set=set_name,
            transform=dataset_transforms
        )

    def _load_weights(self):
        # load last weights
        if self.cfg['load_weights'] is not None:
            if self.cfg['load_weights'].endswith('.pth'):
                weights_path = self.cfg['load_weights']
            else:
                weights_path = get_last_weights(self.cfg['saved_path'])
            try:
                last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
            except:
                last_step = 0

            try:
                ret = self.model.load_state_dict(torch.load(weights_path), strict=False)
            except RuntimeError as e:
                print(f'[Warning] Ignoring {e}')
                print(
                    '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

            print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
        else:
            last_step = 0
            print('[Info] initializing weights...')
            init_weights(self.model)
        return last_step

    def _aspectaware_resize_padding(self, image_shape, width, height):
        old_h, old_w, c = image_shape

        if old_w > old_h:
            new_w = width
            new_h = int(width / old_w * old_h)
        else:
            new_w = int(height / old_h * old_w)
            new_h = height

        return new_w, new_h, old_w, old_h, 0, 280
                
    def save_checkpoint(self, name):
        if isinstance(self.model, CustomDataParallel):
            torch.save(self.model.module.model.state_dict(), os.path.join(self.cfg['save_dir'], name))
        else:
            torch.save(self.model.model.state_dict(), os.path.join(self.cfg['save_dir'], name))
        print("checkpoint saved to ", os.path.join(self.cfg['saved_path'], name))

    def run_cocoeval(self, pred_json_path):

        VAL_GT = os.path.join(self.cfg['data_path'], self.cfg['project_name'], 'annotations/instances_val.json')
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
        
        # prepare coco eval results file
        _, regressions, classifications, anchors = res

        for img, reg, cls, img_meta, img_id in zip(imgs, regressions, classifications, imgs_meta, imgs_ids):
            preds = postprocess(img.unsqueeze(0), anchors, 
                                reg.unsqueeze(0), cls.unsqueeze(0),
                                self.regressBoxes, self.clipBoxes,
                                self.threshold, self.nms_threshold)

            preds = invert_affine([img_meta], preds)[0]
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

    def run_eval(self, epoch, data, writer):

        self.model.eval()
        loss_reg_ls = []
        loss_cls_ls = []
        self.results = []
        max_size = self.input_sizes[self.cfg['compound_coef']]

        progress_bar = tqdm(self.val_generator)
        for iter, data in enumerate(progress_bar):
            with torch.no_grad():
                imgs = data['img']
                annot = data['annot']
                imgs_ids = data['img_id']

                if self.cfg['num_gpus'] == 1:
                    imgs = imgs.cuda()
                    annot = annot.cuda()
                
                imgs_meta = [self._aspectaware_resize_padding(img_shape, max_size, 
                                    max_size) for img_shape in data['orig_shape']]

                cls_loss, reg_loss, res = self.model(imgs, annot, obj_list=self.cfg['obj_list'])
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                loss = cls_loss + reg_loss
                if loss == 0 or not torch.isfinite(loss):
                    continue

                loss_cls_ls.append(cls_loss.item())
                loss_reg_ls.append(reg_loss.item())
                
                if epoch >= 50:
                    self._write_eval_results(imgs, imgs_meta, imgs_ids, res)

        cls_loss = np.mean(loss_cls_ls)
        reg_loss = np.mean(loss_reg_ls)
        loss = cls_loss + reg_loss

        print(
            'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                epoch, self.cfg['num_epochs'], cls_loss, reg_loss, loss))
        writer.add_scalars('Loss', {'val': loss}, self.step)
        writer.add_scalars('Regression_loss', {'val': reg_loss}, self.step)
        writer.add_scalars('Classfication_loss', {'val': cls_loss}, self.step)

        # write output
        if epoch >= 50:
            pred_json_path = os.path.join(self.cfg['saved_path'], 'val_bbox_results.json')
            if os.path.exists(pred_json_path):
                os.remove(pred_json_path)
            json.dump(self.results, open(pred_json_path, 'w'), indent=4)
            print("Prediction json file saved to ", pred_json_path)
        else:
            pred_json_path = None

        return loss, cls_loss, reg_loss, pred_json_path

    def train_one_epoch(self, epoch, last_epoch, writer):

        epoch_loss = []
        progress_bar = tqdm(self.training_generator)
        for iter, data in enumerate(progress_bar):
            if iter < self.step - last_epoch * self.num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                imgs = data['img']
                annot = data['annot']

                if self.cfg['num_gpus'] == 1:
                    # if only one gpu, just send it to cuda:0
                    # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                    imgs = imgs.cuda()
                    annot = annot.cuda()

                self.optimizer.zero_grad()
                cls_loss, reg_loss, res = self.model(imgs, annot, obj_list=self.cfg['obj_list'])
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()

                loss = cls_loss + reg_loss
                if loss == 0 or not torch.isfinite(loss):
                    continue

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                self.optimizer.step()

                epoch_loss.append(float(loss))

                progress_bar.set_description(
                    'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                        self.step, epoch, self.cfg['num_epochs'], iter + 1, self.num_iter_per_epoch, cls_loss.item(),
                        reg_loss.item(), loss.item()))
                writer.add_scalars('Loss', {'train': loss}, self.step)
                writer.add_scalars('Regression_loss', {'train': reg_loss}, self.step)
                writer.add_scalars('Classfication_loss', {'train': cls_loss}, self.step)

                # log learning_rate
                current_lr = self.optimizer.param_groups[0]['lr']
                writer.add_scalar('learning_rate', current_lr, self.step)

                self.step += 1

                if self.step % self.cfg['save_interval'] == 0 and self.step > 0:
                    self.save_checkpoint( f"efficientdet-d{self.cfg['compound_coef']}_{epoch}_{self.step}.pth")
                    print('checkpoint...')

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

        return data, epoch_loss, cls_loss, reg_loss
    
    def collect_common_metrics(self, loss, cls_loss, reg_loss, val_loss, val_cls_loss, val_reg_loss, total_mins):
        return {
            'train_loss': loss,
            'train_cls_loss': cls_loss,
            'train_reg_loss': reg_loss,
            'val_loss': val_loss,
            'val_cls_loss': val_cls_loss,
            'val_reg_loss': val_reg_loss,
            'training_time': total_mins
        }

    def collect_extended_metrics(self, coco_eval):
        return {
            'mAP': coco_eval.stats[0],  # mAP at IoU=0.50:0.95, area=all, maxDets=100
            'mAP_50': coco_eval.stats[1],  # mAP at IoU=0.50, area=all, maxDets=100
            'mAP_75': coco_eval.stats[2],  # mAP at IoU=0.75, area=all, maxDets=100
            'mAP_small': coco_eval.stats[3],  # mAP at IoU=0.50:0.95, area=small, maxDets=100
            'mAP_medium': coco_eval.stats[4],  # mAP at IoU=0.50:0.95, area=medium, maxDets=100
            'mAP_large': coco_eval.stats[5],  # mAP at IoU=0.50:0.95, area=large, maxDets=100
            'aR_maxdet_1': coco_eval.stats[6],  # Precision at IoU=0.50:0.95, area=all, maxDets=100
            'aR_maxdet_10': coco_eval.stats[7],  # Precision at IoU=0.50:0.95, area=all, maxDets=100
            'aR_maxdet_100': coco_eval.stats[8],  # Precision at IoU=0.50:0.95, area=all, maxDets=100
            'aR_small': coco_eval.stats[9],  # Precision at IoU=0.50:0.95, area=small, maxDets=100
            'aR_medium': coco_eval.stats[10],  # Precision at IoU=0.50:0.95, area=medium, maxDets=100
            'aR_large': coco_eval.stats[11]  # Precision at IoU=0.50:0.95, area=large, maxDets=100
        }

def train(**cfg):

    config = {
        "architecture"     : f"EfficientDet-{cfg['compound_coef']}",
        "config"           : cfg['config_file'],
        "compound_coef"    : cfg['compound_coef'], 
        "dataset"          : cfg['project_name'] ,
        "epochs"           : cfg['num_epochs'],
        "pretrain"         : cfg['load_weights'],
        "num_workers"      : cfg['num_workers'],
        "batch_size"       : cfg['batch_size'],
        "num_classes"      : len(cfg['obj_list']),
        "output_dir"       : cfg['save_dir'],
    }

    if cfg['wandb']:
        wandb.init( project=cfg['project'],
                    name = cfg['run_name'],
                    config=config )
        
    if cfg['num_gpus'] == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    cfg['log_path'] = os.path.join(cfg['save_dir'] , "tensorboard")
    os.makedirs(cfg['save_dir'], exist_ok=True)
    os.makedirs(cfg['log_path'], exist_ok=True)
    shutil.copy(cfg['config_file'], cfg['save_dir'])

    trainer = Train_EfficientDet(**cfg)
    trainer.model.train()

    writer = SummaryWriter(cfg['log_path'] + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')
    
    start_training = time.time()

    try:
        for epoch in range(cfg['num_epochs']):

            start_epoch = time.time()
            last_epoch = trainer.step // trainer.num_iter_per_epoch
            if epoch < last_epoch:
                continue

            data, epoch_loss, cls_loss, reg_loss = trainer.train_one_epoch(epoch, last_epoch, writer)
            loss = cls_loss + reg_loss
            
            trainer.scheduler.step(np.mean(epoch_loss))

            if epoch % cfg['val_interval'] == 0:

                val_loss, val_cls_loss, val_reg_loss, pred_json_path = trainer.run_eval(epoch, data, writer)
                
                if not pred_json_path is None:
                    coco_eval = trainer.run_cocoeval(pred_json_path)

                if loss + cfg['es_min_delta'] < trainer.best_loss:
                    trainer.best_loss = loss
                    trainer.best_epoch = epoch

                    trainer.save_checkpoint( f"efficientdet-d{cfg['compound_coef']}_{epoch}_{trainer.step}.pth")

                trainer.model.train()

                # Early stopping
                if epoch - trainer.best_epoch > cfg['es_patience'] > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, trainer.best_loss))
                    break
            
            total_time = time.time() - start_epoch
            total_mins = total_time / 60
            print(f'Epoch {epoch} Training time in mintues {total_mins}')


            metrics = trainer.collect_common_metrics(loss, cls_loss, reg_loss, val_loss, 
                                                     val_cls_loss, val_reg_loss, total_mins)

            if epoch >= 50 and cfg['wandb']:
                metrics.update(trainer.collect_extended_metrics(coco_eval))
                wandb.log(metrics)
    except KeyboardInterrupt:
        trainer.save_checkpoint( f"efficientdet-d{cfg['compound_coef']}_{epoch}_{trainer.step}.pth")
        writer.close()
    writer.close()
    
    total_time = time.time() - start_training
    total_training_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_training_time_str))
    config['training_time'] = total_training_time_str

    # # CONFIG Write to a YAML file
    # with open(os.path.join(cfg['save_dir'],'training_config.yaml'), 'w') as file:
    #     yaml.dump(config, file)
    # print("Outputs and results saved to ", cfg['save_dir'])
    
    if cfg['wandb']:
        wandb.finish()