**DE⫶TR**: End-to-End Object Detection with Transformers
========

First go into the detr path.
```bash
cd models/det_models/detr
```
**Please use the default wandb project name and make sure to specify the wandb run name for each training.** 

Train DETR
```python
cd models/det_models/detr
python main.py --dataset_file adult-kid --data_path ../datasets/adult-kid-v3.1-base-detr-format/ --output_dir exps --resume weights/detr-r50-e632da11.pth --epochs 3 --wandb_run_name detr
```

Resume Training
```python
python main.py --dataset_file adult-kid --data_path ../datasets/adult-kid-v3.1-base-detr-format/ --output_dir output --resume exps/exp1/checkpoint.pth --epochs 101 --start_epoch 100 --wandb_run_name detr
```