# VLP-RVOS

The official implementation of the paper: 

<div align="center">
<h1>
<b>
Harnessing Vision-Language Pretrained Models with Temporal-Aware Adaptation for Referring Video Object Segmentation
</b>
</h1>
</div>


<p align="center"><img src="docs/framework.png" width="800"/></p>

## Introduction

We introduce a framework named VLP-RVOS which harnesses VLP models for RVOS through temporal-aware adaptation. We first propose a temporal-aware prompt-tuning method, which not only adapts pretrained representations for pixel level prediction but also empowers the vision encoder to model temporal contexts. We further customize a cube-frame attention mechanism for robust spatial-temporal reasoning. Besides, we propose to perform multi-stage VL relation modeling while and after feature extraction for comprehensive VL understanding. Extensive experiments demonstrate that our method performs favorably against state-of-the-art algorithms and exhibits strong generalization abilities.

# Installation

## Setup

The main setup of our code follows [Referformer](https://github.com/wjn922/ReferFormer).

First, clone the repository locally.

```
git clone https://github.com/xwt909090/VLP-RVOS
```

Then, install Pytorch==1.11.0 (CUDA 11.3) torchvision==0.12.0 and the necessary packages as well as pycocotools.
```
pip install -r requirements.txt 
```

Finally, compile CUDA operators.
```
cd models/ops
python setup.py build install
cd ../..
```

Please refer to [Referformer](https://github.com/wjn922/ReferFormer) for data preparation.

## Training and Evaluation

All the models are trained using 4 RTX 3090 GPU. 

If you want to train/evaluate VLP-RVOS, please run the following command:

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 --use_env train.py --experiment_name=clip-base
```

```
python test.py --test_start=11 --test_end=11 --ngpu=4 --experiment_name=clip-base
```

Note: You can modify the `--experiment_name` and to specify a config file.

### Results (Ref-Youtube-VOS & Ref-DAVIS)

To evaluate the results, please upload the zip file to the [competition server](https://codalab.lisn.upsaclay.fr/competitions/3282#participate).

## Trained on Ref-Youtube-VOS alone

| Backbone| Ref-YouTube-VOS J&F | Ref-DAVIS J&F |
| :----: | :----: | :----: |
| CLIP(ViT-B/16) | 59.7 | 60.3 |
| VLMo(VLMo-B) | 60.1 | 61.2 |

## Pretrained on Ref-COCO/+/g and fine-tuned on Ref-Youtube-VOS

| Backbone| Ref-YouTube-VOS J&F | Ref-DAVIS J&F |
| :----: | :----: | :----: |
| CLIP(ViT-B/16) | 62.9 | 65.1 |
| VLMo(VLMo-B) | 63.1 | 65.5 |
| CLIP(ViT-L/14) | 66.0 | 68.2 |
| VLMo(VLMo-L) | 67.6 | 70.2 |

### Results (MEVIS)

## Trained on MEVIS alone

| Backbone| MEVIS J&F |
| :----: | :----: |
| CLIP(ViT-B/16) | 44.6 |
| VLMo(VLMo-B) | 45.4 |

### Results (A2D-Sentences & JHMDB-Sentences)

| Backbone | (A2D) mAP | Mean IoU | Overall IoU | (JHMDB) mAP | Mean IoU | Overall IoU |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| CLIP(ViT-B/16) | 53.3 | 69.5 | 76.7 | 44.2 | 71.9 | 73.6 |
| VLMo(VLMo-B) | 53.9 | 72.7 | 78.5 | 44.6 | 72.3 | 73.7 |
| CLIP(ViT-L/14) | 59.4 | 75.3 | 84.0 | 46.0 | 75.9 | 77.9 |
| VLMo(VLMo-L) | 63.1 | 77.7 | 86.2 | 47.1 | 76.6 | 78.3 |

## Acknowledgements

- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [ReferFormer](https://github.com/wjn922/ReferFormer)
- [MTTR](https://github.com/mttr2021/MTTR)

## Contact
If you have any questions about this project, please to contact 1050975744@qq.com.
