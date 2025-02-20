## 
This repository represents the official implementation of the ECCV2022 paper:

### [Dynamic 3D Scene Analysis by Point Cloud Accumulation](http://arxiv.org/abs/2207.12394)

[Shengyu Huang](https://shengyuh.github.io), [Zan Gojcic](https://zgojcic.github.io/), [Jiahui Huang](https://cg.cs.tsinghua.edu.cn/people/~huangjh/), [Andreas Wieser](https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html), [Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html)\
| [ETH Zurich](https://igp.ethz.ch/) | [NVIDIA Toronto AI Lab](https://nv-tlabs.github.io) | [BRCist](https://www.bnrist.tsinghua.edu.cn/) |

<image src="assets/teaser.jpg"/>

### Contact
If you have any questions, please let me know: 
- Shengyu Huang {shengyu.huang@geod.baug.ethz.ch}


### Instructions
This code has been tested on:
- Python 3.10.4, PyTorch 1.12.0+cu116, CUDA 11.6, gcc 11.2.0, GeForce RTX 3090
- Python 3.8.3, PyTorch 1.10.2+cu111, CUDA 11.1, gcc 9.4.0, GeForce RTX 3090

#### Requirements
Please adjust according to your cuda version, then run the following to create a virtual environment:
```shell
virtualenv venv_pcaccumulation
source venv_pcaccumulation/bin/activate
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install pyfilter nestargs
```

Then clone our repository by running:
```shell
git clone https://github.com/prs-eth/PCAccumulation.git
cd PCAccumulation
```

#### Datasets and pretrained models
We provide preprocessed Waymo and nuScenes datasets. The preprocessed dataset and checkpoint can be downloaded by running:
```shell
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gseg/PCAccumulation/data.zip
unzip data.zip
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gseg/PCAccumulation/checkpoints.zip
unzip checkpoints.zip
```
****
### Evaluation
#### Val
To quickly run a sanity check of the data, code, and checkpoints on validation split, please run
```shell
python main.py configs/waymo/waymo.yaml 10 1 --misc.mode=val --misc.pretrain=checkpoints/waymo.pth --path.dataset_base_local=$YOUR_DATASET_FOLDER
```
or 
```shell
python main.py configs/nuscene/nuscene.yaml 10 1 --misc.mode=val --misc.pretrain=checkpoints/nuscene.pth --path.dataset_base_local=$YOUR_DATASET_FOLDER
```
You will see the evaluation metrics like the following:
```shell
Successfully load pretrained model from checkpoints/nuscene.pth at epoch 77!
Current best loss 1.3937173217204215
Current best metric 0.8779626780515821
val Epoch: 0	mos_iou: 0.880	mos_recall: 0.942	mos_precision: 0.930	fb_iou: 0.856	fb_recall: 0.918	fb_precision: 0.918	ego_l1_loss: 0.161	ego_l2_loss: 0.119	ego_rot_error: 0.227	ego_trans_error: 0.100	perm_loss: 0.010	fb_loss: 0.341	mos_loss: 0.401	offset_loss: 0.329	offset_l1_loss: 0.531	offset_dir_loss: 0.127	offset_l2_error: 0.436	obj_loss: 0.139	inst_l2_error: 0.214	dynamic_inst_l2_error: 0.268	loss: 1.378	
static:  IoU: 0.929,  Recall: 0.954,  Precision: 0.972 
dynamic:  IoU: 0.832,  Recall: 0.93,  Precision: 0.887 
background:  IoU: 0.974,  Recall: 0.987,  Precision: 0.987 
foreground:  IoU: 0.737,  Recall: 0.849,  Precision: 0.849 
```
in ```snapshot/nuscene/log```

#### Test
To evaluate on the held-out test set, please run
```shell
python main.py configs/waymo/waymo.yaml 1 1 --misc.mode=test --misc.pretrain=checkpoints/waymo.pth --path.dataset_base_local=$YOUR_DATASET_FOLDER
```
This will save per-scene flow estimation/errors to ```results/waymo```. Next, please run the following script to get final evaluation:
```shell
python toolbox/evaluation.py results/waymo waymo
```


### Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@inproceedings{huang2022accumulation,
  title={Dynamic 3D Scene Analysis by Point Cloud Accumulation},
  author={Shengyu Huang and Zan Gojcic and Jiahui Huang and Andreas Wieser, Konrad Schindler},
  booktitle={European Conference on Computer Vision, ECCV},
  year={2022}
}
```

### Acknowledgements
In this project we use (parts of) the following repositories:
- [ConvolutionalOccupancyNetwork](https://github.com/autonomousvision/convolutional_occupancy_networks) and [MotionNet](https://github.com/pxiangwu/MotionNet) for our backbone.
- [TorchSparse](https://github.com/mit-han-lab/torchsparse) and [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) for efficient voxelisation and scatter operations
- [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) for tracking baseline
- [ChamferDistance](https://github.com/chrdiller/pyTorchChamferDistance)

We thank the respective developers for open sourcing and maintenance. We would also like to thank reviewers 1 & 2 for their valuable inputs.
