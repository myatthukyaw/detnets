mkdir -p weights/yolo
mkdir -p weights/efficient-det
mkdir -p weights/detr

wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth 
wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth
wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth
wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth
wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth

wget -P weights/efficient-det https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
wget -P weights/efficient-det https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth
wget -P weights/efficient-det https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth
wget -P weights/efficient-det https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth
