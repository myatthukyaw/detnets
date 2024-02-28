# DetNets - SOTA Detection Networks


Welcome to the DetNets repository, which contains a collection of SOTA object detection models and streamlined for simplified training, evaluation, and inference processes. 

While repositories like Ultralytics' YOLOv8 offer comprehensive features and extensive documentation, we understand that navigating and reproducing results from various sources can sometimes be challenging. For cases where original repositories pose reproducibility issues or are complex to test, DetNets serves as an accessible alternative.

We've refined the training, validation, and inference procedures to adhere to a standardized format, making it straightforward for users to implement these powerful models in their projects. We've also integrated Weights & Biases (WandB) to enhance the visibility and comparability of model training metrics. For in-depth insights and features, we encourage referencing the original repositories. DetNets is here to make your journey in object detection smoother and more efficient.

Available SOTA Detection Models
 
- [x] [EfficientDet](https://github.com/myatthukyaw/detnets/tree/main/models/efficient-det) - [Original Repository](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- [x] [Yolov8](https://github.com/myatthukyaw/detnets/tree/main/models/ultralytics) - [Original Repository](https://github.com/ultralytics/ultralytics)
- [x] [Yolov5](https://github.com/myatthukyaw/detnets/tree/main/models/ultralytics) - [Original Repository](https://github.com/ultralytics/ultralytics)
- [ ] [DETR](https://github.com/myatthukyaw/detnets/blob/main/models/detr) - [Original Repository](https://github.com/facebookresearch/detr)
- [x] [RTDETR](https://github.com/myatthukyaw/detnets/tree/main/models/ultralytics) - [Original Repository](https://github.com/ultralytics/ultralytics)
- [ ] [Detectron2](https://github.com/myatthukyaw/detnets/tree/main/models/detectron2) - [Original Repository](https://github.com/facebookresearch/detectron2)



## Installation

```bash
# Clone this repository
git clone https://github.com/myatthukyaw/detnets.git
cd detnets

# Install dependencies (ensure you meet the prerequisites)
pip install -r requirements.txt
```

## How to Use

Step 1: Preparing Your Data
Prepare your data and put it in some folers. There are two general formats for training the detection models : yolo and coco. 
Select the format depending on the model you choose. For example, you need to prepare your dataset in yolo format for yolov5 and v8 models.
We provide scripts for converting the datasets between yolo and coco formats. 

```python
# convert yolo format dataset to coco
python scripts/yolo2coco.py
# convert coco format dataset to yolo
python scripts/coco2yolo.py
```

Step 2: Selecting a Model
Choose a model that fits your requirements and update the training configuration file under [configs](https://github.com/myatthukyaw/detnets/configs) folder.

Step 3: Download the pretrained weights

```bash
chmod +x scripts/download_weights.sh
./scripts/download_weights.sh
```

Step 4: One script to run all.
You can run all tasks for all models using our [main](https://github.com/myatthukyaw/detnets/main.py) script.
There are two arguments to specify. 
- model (yolo, efficient-det, detr)
- task (train, val, efficient)

Step 3: Training
```python
python train.py --model yolo --task train
```

Step 4: Evaluation
```python
python train.py --model yolo --task val
```

Step 5: Inference
```python
python train.py --model yolo --task inference
```

## Model Comparison
We will provide the detailed benchmark results for all models soon.

For detailed benchmarks, please refer to [Benchmarks.md](BENCHMARKS.md).


## How to Contribute
We welcome contributions from the community! Please see our [Contribution Guidelines](CONTRIBUTION.md) for more information on how you can get involved.

## Acknowledgments
This project builds upon the hard work and contributions of many researchers and developers. We aim to credit all sources appropriately and encourage users to refer to the original works for in-depth understanding. For specific model attributions, please see the [Attributions section](ATTRIBUTION.md).

## License
This repository is licensed under [MIT License](LICENSE). Note that individual models and their associated software may carry their own licenses.