<h1 align="center">
KeyMatchNet: Zero-Shot Pose Estimation in 3D Point Clouds by Generalized Keypoint Matching
</h1>

<div align="center">
<a href="https://keymatchnet.github.io/">[Project Page]</a>
</div>

To test KeyMatchNet on your own data:

>		pip install git+https://github.com/fhagelskjaer/keymatchnet.git
>		keymatchnet --obj test_object.stl --scene point_cloud.pcd --visu True

To download and test KeyMatchNet:

>		git clone https://github.com/fhagelskjaer/keymatchnet.git
>		cd keymatchnet
>		pip install .
>		python -m keymatchnet.single_pe_keymatchnet --model_root trained_network/keymatchnet_electronics/models/model.t7 --obj testdata/09_BGPSL6-9-L30-F7.stl --scene testdata/point_cloud.pcd --visu True

To test the code on the real test data. Download mfe.zip from https://nextcloud.sdu.dk/index.php/s/YgMkA3eHagRjd5k and place it in the current folder and unzip.

>		python -m keymatchnet.pe_keymatchnet --dataset_name picklecap --model_root trained_network/keymatchnet_electronics/models/model.t7
>		python -m keymatchnet.pe_keymatchnet --dataset_name pickleblu --model_root trained_network/keymatchnet_electronics/models/model.t7

The experiments should give approximately 0.74 and 0.73, respectively.

# Installation

Versions used in paper: 

Python 3.9.15

>		pip install numpy==1.24.2
>		pip install torch==2.1.2
>		pip install open3d==0.18.0
>		pip install trimesh==3.18.1
>		pip install distinctipy==1.3.4
>		pip install h5py==3.7.0

# Citation
If you use this code in your research, please cite the paper:

```
@article{hagelskjaer2023keymatchnet,
  title={Keymatchnet: Zero-shot pose estimation in 3d point clouds by generalized keypoint matching},
  author={Hagelskj{\ae}r, Frederik and Haugaard, Rasmus Laurvig},
  journal={arXiv preprint arXiv:2303.16102},
  year={2023}
}
```
