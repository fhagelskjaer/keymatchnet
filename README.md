<h1 align="center">
KeyMatchNet: Zero-Shot Pose Estimation in 3D Point Clouds by Generalized Keypoint Matching
</h1>

<div align="center">
<a href="https://keymatchnet.github.io/">[Project Page]</a>
</div>

To test the code on the real dataset:

>		git clone https://github.com/fhagelskjaer/keymatchnet.git
>		cd keymatchnet

Download mfe.zip place in current folder and unzip.

>		cd keymathcnet
>		python pe_keymatchnet.py --dataset_name picklecap --model_root trained_network/keymatchnet_electronics/models/model.t7
>		python pe_keymatchnet.py --dataset_name pickleblu --model_root trained_network/keymatchnet_electronics/models/model.t7

The experiments should give approximately 0.74 and 0.73, respectively.

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
