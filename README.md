# where it from
* yolov8 forked from [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* mamba-yolo forked from [HZAI-ZJNU/Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)

# install

1. clone this repo
```shell
git clone https://github.com/StanleySun233/mamba-yolo.git
cd mamba-yolo
```

2. note the environment:
* python=3.9
* cuda>=12.6
* torch=2.3.0

```shell
pip install torch===2.3.0 torchvision torchaudio
pip install seaborn thop timm einops
pip install opencv-python
pip install timm
pip install scipy
cd selective_scan
pip install .
cd ..
```

3. run test demo for mamba
```shell
python mbyolo_train.py --task train --data ultralytics/cfg/datasets/coco8.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B.yaml --amp \
  --project ./output_dir/mscoco --name mambayolo_n
```