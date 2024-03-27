import os
import yaml
import argparse
import datetime
import json
import random
import time
import wandb
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='adultkid')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--data_path', type=str, default='../datasets/adult-kid-v3.1-base-detr-format/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=13, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument("--wandb_project", default="detection-baselines" ,help="training output path")
    parser.add_argument("--wandb_run_name", default="test_run" ,help="training output path")
    parser.add_argument("--train_machine_name", default="foundation_prep" ,help="training output path")

    return parser

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

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    args.output_dir = increment_path(args.output_dir, sep='exp', mkdir=True)
    config = {
        "architecture": "DETR",
        "dataset": args.data_path,
        "backbone" : args.backbone,
        "dilation" : args.dilation,
        "epochs": args.epochs,
        "weight" : args.resume,
        "num_workers" : args.num_workers,
        "batch_size" : args.batch_size,
        "start_epoch" : args.start_epoch,
        "lr" : args.lr,
        "input_format" : args.lr,
        "output_dir" : args.output_dir,
        "training_script" : "models/det_models/detr/main.py",
        "training_machine" : args.train_machine_name,
        }

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project = args.wandb_project,
        name = args.wandb_run_name,
        # track hyperparameters and run metadata
        config=config
    )

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        start_epoch = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator, coco_eval = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
        
        total_time = time.time() - start_epoch
        total_epoch_mins = total_time / 60
        # get eval metrcs 
        mAP = coco_eval.stats[0]  # mAP at IoU=0.50:0.95, area=all, maxDets=100
        mAP_50 = coco_eval.stats[1]  # mAP at IoU=0.50, area=all, maxDets=100
        mAP_75 = coco_eval.stats[2]  # mAP at IoU=0.75, area=all, maxDets=100
        mAP_small = coco_eval.stats[3]  # mAP at IoU=0.50:0.95, area=small, maxDets=100
        mAP_medium = coco_eval.stats[4]  # mAP at IoU=0.50:0.95, area=medium, maxDets=100
        mAP_large = coco_eval.stats[5]  # mAP at IoU=0.50:0.95, area=large, maxDets=100
        aR_maxdet_1 = coco_eval.stats[6]  # Precision at IoU=0.50:0.95, area=all, maxDets=100
        aR_maxdet_10 = coco_eval.stats[7]  # Precision at IoU=0.50:0.95, area=all, maxDets=100
        aR_maxdet_100 = coco_eval.stats[8]  # Precision at IoU=0.50:0.95, area=all, maxDets=100
        aR_small = coco_eval.stats[9]  # Precision at IoU=0.50:0.95, area=all, maxDets=100
        aR_medium = coco_eval.stats[10]  # Precision at IoU=0.50:0.95, area=all, maxDets=100
        aR_large = coco_eval.stats[11]  # Precision at IoU=0.50:0.95, area=all, maxDets=100

        metrics = { 'mAP' : mAP,
                    'mAP_50' : mAP_50,
                    'mAP_75' : mAP_75,
                    'mAP_small' : mAP_small,
                    'mAP_medium' : mAP_medium,
                    'mAP_large' : mAP_large,
                    'aR_maxdet_1' : aR_maxdet_1,
                    'aR_maxdet_10' : aR_maxdet_10,
                    'aR_maxdet_100' : aR_maxdet_100,
                    'aR_small' : aR_small,
                    'aR_medium' : aR_medium,
                    'aR_large' : aR_large,
                    'train_lr': log_stats['train_lr'],
                    'train_class_error' : log_stats['train_class_error'],
                    'train_loss' : log_stats['train_loss'],
                    'train_ce_loss' : log_stats['train_loss_ce'],
                    'train_box_loss' : log_stats['train_loss_bbox'],
                    'train_giou_loss' : log_stats['train_loss_giou'],
                    'val_class_error' : log_stats['test_class_error'],
                    'val_loss' : log_stats['test_loss'],
                    'val_ce_loss' : log_stats['test_loss_ce'],
                    'val_box_loss' : log_stats['test_loss_bbox'],
                    'val_giou_loss' : log_stats['test_loss_giou'],
                    'training_time' : total_epoch_mins}
        wandb.log(metrics)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    config["training_time"] = total_time_str
    with open(os.path.join(args.output_dir,'training_config.yaml'), 'w') as file:
        yaml.dump(config, file)
    print("Outputs and results saved to ", args.output_dir)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
