nohup python train.py -c 0 -p adult-kid --batch_size 20 --lr 1e-5 --num_workers 13 --num_epochs 200 --load_weights weights/efficientdet-d0.pth --train_machine_name foundation-pred --data_path ../../../../datasets/
nohup python train.py -c 1 -p adult-kid --batch_size 11 --lr 1e-5 --num_workers 13 --num_epochs 200 --load_weights weights/efficientdet-d1.pth --train_machine_name foundation-pred --data_path ../../../../datasets/

python coco_eval.py -p adult-kid -w logs/adult-kid-v3.1-base-coco-format/efficientdet-d1_99_48000.pth -c 1