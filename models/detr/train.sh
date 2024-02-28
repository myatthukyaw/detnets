nohup python main.py --dataset_file adult-kid --data_path ../datasets/adult-kid-v3.1-base-coco-format/ --output_dir output --resume weights/detr-r50-e632da11.pth --epochs 30

resume 
nohup python main.py --dataset_file adult-kid --data_path ../datasets/adult-kid-v3.1-base-coco-format/ --output_dir output --resume output/checkpoint.pth --epochs 101 --start_epoch 100

to test
nohup python main.py --dataset_file adult-kid --data_path ../datasets/adult-kid-v3.1-base-coco-format/ --output_dir output/num_queries_50_train --resume output/checkpoint.pth --epochs 100 