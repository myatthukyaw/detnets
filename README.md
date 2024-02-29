# DetNets - SOTA Detection Networks


Welcome to the DetNets repository, which contains a collection of SOTA object detection models and streamlined for simplified training, evaluation, and inference processes. 

While repositories like Ultralytics' YOLOv8 offer comprehensive features and extensive documentation, we understand that navigating and reproducing results from various sources can sometimes be challenging for cases where original repositories pose reproducibility issues or are complex to test, DetNets serves as an accessible alternative.

We've refined the training, validation, and inference procedures to adhere to a standardized format, making it straightforward for users to implement these powerful models in their projects. We've also integrated Weights & Biases (WandB) to enhance the visibility and comparability of model training metrics. For in-depth insights and features, we encourage referencing the original repositories. DetNets is here to make your journey in object detection smoother and more efficient.

Available SOTA Detection Models

- [x] [Yolov8](https://github.com/myatthukyaw/detnets/tree/main/models/ultralytics) - [Original Repository](https://github.com/ultralytics/ultralytics)
- [x] [Yolov5](https://github.com/myatthukyaw/detnets/tree/main/models/ultralytics) - [Original Repository](https://github.com/ultralytics/ultralytics)
- [x] [RTDETR](https://github.com/myatthukyaw/detnets/tree/main/models/ultralytics) - [Original Repository](https://github.com/ultralytics/ultralytics)
- [ ] [EfficientDet](https://github.com/myatthukyaw/detnets/tree/main/models/efficient-det) - [Original Repository](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- [ ] [DETR](https://github.com/myatthukyaw/detnets/blob/main/models/detr) - [Original Repository](https://github.com/facebookresearch/detr)
<!-- - [ ] [Detectron2](https://github.com/myatthukyaw/detnets/tree/main/models/detectron2) - [Original Repository](https://github.com/facebookresearch/detectron2) -->



## Installation

```bash
# Clone this repository
git clone https://github.com/myatthukyaw/detnets.git
cd detnets

# Install dependencies (ensure you meet the prerequisites)
pip install -r requirements.txt
```

## How to Use

Step 1: Preparing Your Data</br>
There are two dataset formats for training the detection models : yolo and coco. Prepare your dataset in one of those format according to your model accepted format. You can convert your dataset format between yolo and coco using our provided scripts. 

```python
# convert yolo format dataset to coco
python scripts/yolo2coco.py --dataset_root datasets/my_dataset_yolo --output_dataset datasets/my_dataset_coco
# convert coco format dataset to yolo
python scripts/coco2yolo.py --coco_dataset_root datasets/my_dataset_coco --output_yolo_dataset datasets/my_dataset_yolo
```

Step 2: Selecting a Model</br>
Choose a model that fits your requirements and update the training configuration file under [configs](https://github.com/myatthukyaw/detnets/blob/main/configs) folder.

<table>
  <tr>
    <th>Model</th>
    <th>Base</th>
    <th>Configuration File</th>
  </tr>
  <tr>
    <td>YOLOv8, YOLOv5</td>
    <td>YOLO</td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/yolo.yml">config</a></td>
  </tr>
  <tr>
    <td>RTDETR</td>
    <td>YOLO</td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/rtdetr.yml">config</a></td>
  </tr>
  <tr>
    <td>DETR</td>
    <td>COCO</td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/detr.yml">config</a></td>
  </tr>
  <tr>
    <td>Efficient-Det</td>
    <td>COCO</td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/efficient-det.yml">config</a></td>
  </tr>
</table>

Step 3: Download the pretrained weights

```bash
chmod +x scripts/download_weights.sh
./scripts/download_weights.sh
```

Step 4: One script to run them all.</br>
You can run all tasks for all models using our [main](https://github.com/myatthukyaw/detnets/blob/main/main.py) script.
There are two arguments to specify. 
- model (yolo, efficient-det, detr)
- task (train, val, efficient)

Step 5: Training
```python
python train.py --model yolo --task train
```

Step 6: Evaluation
```python
python train.py --model yolo --task val
```

Step 7: Inference
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