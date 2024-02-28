# Detectron2
========
## Train Detectron2 Models

Please check the original detectron2 repo for installation and more info. 
https://github.com/facebookresearch/detectron2

Train Detectron2 on our dataset
```python
python -m models.det_models.detectron2.train_custom_ds \
                    --config COCO-Detection/retinanet_R_50_FPN_1x.yaml   \              
                    --dataset_dir ../datasets/adult-kid-v3.1-base-coco-format/       \                      
                    --batch_size 7                 \
                    --num_workers 13                 \
                    --iterations 10                 \
                    --output_dir models/det_models/detectron2/exps                \ 
                    --wandb_project detection-baselines                 \
                    --wandb_run_name test_run
```