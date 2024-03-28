mkdir -p weights/yolov7
mkdir -p weights/yolov9
mkdir -p weights/efficient-det
mkdir -p weights/rt-detr

wget -P weights/rt-detr https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt
wget -P weights/rt-detr https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-x.pt

wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth 
wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth
wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth
wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth
wget -P weights/efficient-det https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth

wget -P weights/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
wget -P weights/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
wget -P weights/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt
wget -P weights/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
wget -P weights/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt
wget -P weights/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt

wget -P weights/yolov9 https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
wget -P weights/yolov9 https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt

#wget -P weights/efficient-det https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
#wget -P weights/efficient-det https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth
#wget -P weights/efficient-det https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth
#wget -P weights/efficient-det https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth
