# DetNets - SOTA Detection Networks


Welcome to the DetNets repository, which contains a collection of SOTA object detection models and streamlined for simplified training, evaluation, and inference processes. 

While some repositories offer comprehensive features and extensive documentation, we understand that navigating and reproducing results from various sources can sometimes be challenging for cases where original repositories pose reproducibility issues or are complex to test, DetNets serves as an accessible alternative.

We've refined the training, validation, and inference procedures to adhere to a standardized format, making it straightforward for users to implement these powerful models in their projects. We've also integrated Weights & Biases (WandB) to enhance the visibility and comparability of model training metrics and logs. For in-depth insights and features, we encourage referencing the original repositories. DetNets is here to make your journey in object detection smoother and more efficient.

<a id="models">Available SOTA Detection Models and modes/tasks</a>

<table>
  <tr>
    <th>Model</th>
    <th>Original Repository</th>
    <th>Config</th>
    <th>Train</th>
    <th>Evaluation</th>
    <th>Inference</th>
  </tr>
  <tr>
    <td>YOLOv8</td>
    <td><a href="https://github.com/ultralytics/ultralytics">Ultralytics</a></td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/yolo.yml">yolo.yml</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <tr>
    <td>YOLOv5</td>
    <td><a href="https://github.com/ultralytics/ultralytics">Ultralytics</a></td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/yolo.yml">yolo.yml</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <tr>
    <td>RTDETR</td>
    <td><a href="https://github.com/ultralytics/ultralytics">Ultralytics</a></td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/rtdetr.yml">rtdetr.yml</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Efficient-Det</td>
    <td><a href="https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch">Yet-Another-EfficientDet-Pytorch</a></td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/efficient-det.yml">efficient-det.yml</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>DETR</td>
    <td><a href="https://github.com/facebookresearch/detr">DETR</a></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>SSD</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>YOLOv9</td>
    <td><a href="https://github.com/WongKinYiu/yolov9">yolov9</a></td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/yolov9.yml">yolov9.yml</a></td>
    <td>-</td>
    <td>-</td>
    <td>✅</td>
  </tr>
    <tr>
    <td>YOLOv7</td>
    <td><a href="https://github.com/WongKinYiu/yolov7">yolov7</a></td>
    <td><a href="https://github.com/myatthukyaw/detnets/blob/main/configs/yolov7.yml">yolov7.yml</a></td>
    <td>-</td>
    <td>-</td>
    <td>✅</td>
  </tr>
    <tr>
    <td>YOLOR</td>
    <td><a href="https://github.com/WongKinYiu/yolor">yolor</a></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>YOLOX</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>



## Installation

Create a conda environment
```bash
conda create --name detnets python=3.8
```

Install Pytorch. 
Pytorch 1.10.0 works for all models on most GPU types but in A100 GPU, yolo models might have some issues relating CUDA. Try installing Pytorch 2 for yolo models in separate environment if CUDA issues occurs.
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Clone the repo and install dependencies
```bash
# Clone this repository
git clone https://github.com/myatthukyaw/detnets.git
cd detnets

# Install dependencies (ensure you meet the prerequisites)
pip install -r requirements.txt --no-cache-dir 
```

## How to Use

#### Step 1: Preparing Your Data</br>
Training detection models requires datasets in one of two formats: YOLO or COCO. Ensure your dataset is prepared in the format compatible with your chosen model. We offer scripts that allow you to easily convert your dataset between the YOLO and COCO formats.


```python
# convert yolo format dataset to coco
python scripts/yolo2coco.py --dataset_root datasets/my_dataset_yolo --output_dataset datasets/my_dataset_coco
# convert coco format dataset to yolo
python scripts/coco2yolo.py --coco_dataset_root datasets/my_dataset_coco --output_yolo_dataset datasets/my_dataset_yolo
```

#### Step 2: Selecting a Model</br>
Select a model that aligns with your project needs and update the corresponding training configuration file located in the [configs](https://github.com/myatthukyaw/detnets/blob/main/configs) directory.

Refer to the [table](#models) above to review the configuration file for each model.

General configurations, such as the dataset directory and Weights & Biases (W&B) logging settings, are located in the initial section of the configuration file. Specific parameters for training, testing, and inference tasks can be found in their respective sections within each YAML configuration file.

#### Step 3: Download the pretrained weights

```bash
chmod +x scripts/download_weights.sh
./scripts/download_weights.sh
```

#### Step 4 : Configure WandB credentials 

If you don't want to log the training metrics to W&B, set the W&B flag to False in the config file. 

Otherwise, configure W&B. 

```bash
wandb login 
# enter your API key
```

#### Step 5: One script to run them all.</br>
You can run all tasks for all models using our [main](https://github.com/myatthukyaw/detnets/blob/main/main.py) script.
There are two arguments to specify. 
- model (yolov8, yolov5, rt-detr, yolov7, yolov9, efficient-det, detr)
- mode (train, test, inference)
Lets go to next step for more information. 

#### Step 6: Training and Evaluation
```python
# training
python main.py --model yolo --mode train

# evaluation
python main.py --model yolo --mode test
```

#### Step 7 : Inference

Inference parameters can be configure in each model yaml configuration file.

Here is an example:
```bash
...
# inference configuration
inference
  weight      : yolov8n.pt       # your trained weight or sth
  source      : data/demo.jpg        # image or video to run inference
  conf_thres  : 0.5
  nms_thres   : 0.5
  show        : False
  save        : True
...
```

```python
# inference
python train.py --model yolo --mode inference
```

#### Step 8 : Saving the output artifacts for each task.

Everytime you run the train, val and inference tasks, the results artifacts like training config, trained model, result metrics, and images, will be saved in a directory.
The directory format is as below: 
```bash
# for training and testing
runs/{project}/{model}/{run-name}-exp{x}

# for inference
runs/{project}/{model}/outputs-{x}
```

- **project** - dataset name is recommaned to use for this. If wandb is true, this name will be the same as the project name on wandb. 
- **model** - model you used to train/val/run inference.
- **run_name** - run name you want to give for training task. If wandb is true, this name will be the same as the wandb run. 
- **x** - iterative variable for repetative runs or outputs


## Model Comparison
We will provide the detailed benchmark results for all models soon.

For detailed benchmarks, please refer to [Benchmarks.md](BENCHMARKS.md).


## How to Contribute
We welcome contributions from the community! Please see our [Contribution Guidelines](CONTRIBUTION.md) for more information on how you can get involved.

## Acknowledgments
This project builds upon the hard work and contributions of many researchers and developers. We aim to credit all sources appropriately and encourage users to refer to the original works for in-depth understanding. For specific model attributions, please see the [Attributions section](ATTRIBUTION.md).

## License
This repository is licensed under [MIT License](LICENSE). Note that individual models and their associated software may carry their own licenses.