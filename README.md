# Installation

## for Mac

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" pip install -e detectron2
```

# Download weights

```
wget https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl -O weights/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.pkl
```
