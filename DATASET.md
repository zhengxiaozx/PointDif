## Dataset


### ShapeNet55/34 Dataset:

We follow the [Point-BERT](https://github.com/LilRedWu/PointBert) and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) repos for preprocessing data.

```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  ├── .......
├──ShapeNet-55/
│  ├── train.txt
│  └── test.txt
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). 


### ScanObjectNN Dataset:
We follow the [Point-BERT](https://github.com/LilRedWu/PointBert) and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) repos for preprocessing data.
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).



### ScanNet Dataset:
We follow the [MaskPoint](https://github.com/WisconsinAIVision/3detr_MaskPoint?tab=readme-ov-file) repo for preprocessing data.

```
│ScanNetV2/
├──scans
│  ├── scene0000_00
│  ├── ...
│  ├── scene0706_00
```

Download: Please download the data from the [official website](https://github.com/ScanNet/ScanNet/).


### S3DIS Dataset:
We follow the [PointNeXt](https://github.com/guochengqian/pointnext) repo for preprocessing data.

```
│S3DIS/
├──s3disfull/
│  ├──raw
│     ├── Area_6_pantry_1.npy
│     ├── ...
│  ├──processed
│     ├── s3dis_val_area5_0.040.pkl
```

Download: Please download the data from the [PointNeXt repo](https://guochengqian.github.io/PointNeXt/examples/s3dis/).


### SemanticKITTI Dataset:

We follow the [OpenPCSeg](https://github.com/guochengqian/pointnext) repo for preprocessing data.

```
│SemanticKitti/
├──dataset/
│  ├──velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
│      ├── 00
│      ├── ...
│      ├── 21
│  ├──labels   <- contains the .label files; a .label file contains the labels of the points in a point cloud
│      ├── 00
│      ├── ...
│      ├── 10
│  ├──calib
│      ├── 00
│      ├── ...
│      ├── 21
│  ├──semantic-kitti.yaml
```
Download: Please download the data from the [official website](https://semantic-kitti.org/index).