# Training/model creation

## Installation instructions
We run YOLOv11.
This guide assumes a Ubuntu environment.
It assumes you have a compatible NVIDIA GPU.
Many commands will be similar on other systems.
Look at [PyTorch installation](https://pytorch.org/get-started/locally/) instructions for more info.

Install PyTorch, assuming CUDA 13.0
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Install Ultralytics
```bash
uv pip install -U ultralytics
```

The installation will take up about 5GB of space

## Running the model
1. Prepare the data splitting by SJH subject id using `train_test_split.py`
2. Run the YOLO training: `yolo obb train data=yolo_model_desc.yaml batch=-1 imgsz=512 cache=True`
    - delete any cache files from prior runs