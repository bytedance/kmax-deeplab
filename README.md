# kMaX-DeepLab (ECCV 2022)

This is a *PyTorch re-implementation* of our ECCV 2022 paper based on Detectron2: [k-means mask Transformer](https://arxiv.org/pdf/2207.04044.pdf).

*Disclaimer*: This is a *re-implementation* of kMaX-DeepLab in PyTorch. While we have tried our best to reproduce all the numbers reported in the paper, please refer to the original numbers in the [paper](https://arxiv.org/pdf/2207.04044.pdf) or [tensorflow repo](https://github.com/google-research/deeplab2/blob/main/g3doc/projects/kmax_deeplab.md) when making performance or speed comparisons.

[kMaX-DeepLab](https://arxiv.org/pdf/2207.04044.pdf) is an end-to-end method for
general segmentation tasks. Built upon
[MaX-DeepLab](https://arxiv.org/pdf/2012.00759.pdf) and
[CMT-DeepLab](https://arxiv.org/pdf/2206.08948.pdf), kMaX-DeepLab proposes a
novel view to regard the mask transformer as a process of iteratively
performing cluster-assignment and cluster-update steps.

<p align="center">
   <img src="./docs/clustering_view_of_mask_transformer.png" width=450>
</p>

Insipred by the similarity between cross-attention and k-means clustering
algorithm, kMaX-DeepLab proposes k-means cross-attention, which adopts a simple
modification by changing the activation function in cross-attention from
spatial-wise softmax to cluster-wise argmax.

<p align="center">
   <img src="./docs/kmax_decoder.png" width=500>
</p>

As a result, kMaX-DeepLab not only produces much more plausible attention map
but also enjoys a much better performance.


## Installation
The code-base is verified with pytorch==1.12.1, torchvision==0.13.1, cudatoolkit==11.3, and detectron2==0.6,
please install other libiaries through *pip3 install -r requirements.txt*

Please refer to [Mask2Former's script](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md) for data preparation.


## Model Zoo
Note that model zoo below are *trained from scratch using this PyTorch code-base*, we also offer code for porting and evaluating the [TensorFlow checkpoints](https://github.com/google-research/deeplab2/blob/main/g3doc/projects/kmax_deeplab.md) in the section *Porting TensorFlow Weights*.

### COCO Panoptic Segmentation
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">PQ<sup>thing</sup></th>
<th valign="bottom">PQ<sup>stuff</sup></th>
<th valign="bottom">ckpt</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="./configs/coco/panoptic_segmentation/kmax_r50.yaml">ResNet-50</td>
<td align="center"> 53.3 </td>
<td align="center"> 83.2 </td>
<td align="center"> 63.3 </td>
<td align="center"> 58.8 </td>
<td align="center"> 45.0 </td>
<td align="center"><a href="https://drive.google.com/file/d/1YB_5dct0U7ys2KTJNjDIqXLSZneWTyr-/view?usp=sharing">download</a></td>
</tr>
 <tr><td align="left"><a href="./configs/coco/panoptic_segmentation/kmax_convnext_tiny.yaml">ConvNeXt-Tiny</a></td>
<td align="center"> 55.5 </td>
<td align="center"> 83.3 </td>
<td align="center"> 65.9 </td>
<td align="center"> 61.4 </td>
<td align="center"> 46.7 </td>
<td align="center"><a href="https://drive.google.com/file/d/1KAEztHbVG3Pvi6JnrCMtRYTgSFi7zr47/view?usp=drive_link">download</a></td>
</tr>
 <tr><td align="left"><a href="./configs/coco/panoptic_segmentation/kmax_convnext_small.yaml">ConvNeXt-Small</a></td>
<td align="center"> 56.7 </td>
<td align="center"> 83.4 </td>
<td align="center"> 67.2 </td>
<td align="center"> 62.7 </td>
<td align="center"> 47.7 </td>
<td align="center"><a href="https://drive.google.com/file/d/1yRmGWrpUyXCL-QgAm00tRU981RhX2gG2/view?usp=sharing">download</a></td>
</tr>
 <tr><td align="left"><a href="./configs/coco/panoptic_segmentation/kmax_convnext_base.yaml">ConvNeXt-Base</a></td>
<td align="center"> 57.2 </td>
<td align="center"> 83.4 </td>
<td align="center"> 67.9 </td>
<td align="center"> 63.4 </td>
<td align="center"> 47.9 </td>
<td align="center"><a href="https://drive.google.com/file/d/18fWcWxeBw7HuKU-llu0hanBwaVYd7nB4/view?usp=drive_link">download</a></td>
</tr>
 <tr><td align="left"><a href="./configs/coco/panoptic_segmentation/kmax_convnext_large.yaml">ConvNeXt-Large</a></td>
<td align="center"> 57.9 </td>
<td align="center"> 83.5 </td>
<td align="center"> 68.5 </td>
<td align="center"> 64.3 </td>
<td align="center"> 48.4 </td>
<td align="center"><a href="https://drive.google.com/file/d/1b6rEnKw4PNTdqSdWpmb0P9dsvN0pkOiN/view?usp=sharing">download</a></td>
</tr>
</tbody></table>


### Cityscapes Panoptic Segmentation
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">PQ<sup>thing</sup></th>
<th valign="bottom">PQ<sup>stuff</sup></th>
<th valign="bottom">AP</th>
<th valign="bottom">IoU</th>
<th valign="bottom">ckpt</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="./configs/cityscapes/panoptic_segmentation/kmax_r50.yaml">ResNet-50</td>
<td align="center"> 63.5 </td>
<td align="center"> 82.0 </td>
<td align="center"> 76.5 </td>
<td align="center"> 57.8 </td>
<td align="center"> 67.7 </td>
<td align="center"> 38.6 </td>
<td align="center"> 79.5 </td>
<td align="center"><a href="https://drive.google.com/file/d/1v1bsifuF21ft7wMwgjJNSJu5JBowoNta/view?usp=sharing">download</a></td>
 <tr><td align="left"><a href="./configs/cityscapes/panoptic_segmentation/kmax_convnext_large.yaml">ConvNeXt-Large</a></td>
<td align="center"> 68.4 </td>
<td align="center"> 83.3 </td>
<td align="center"> 81.3 </td>
<td align="center"> 62.6 </td>
<td align="center"> 72.6 </td>
<td align="center"> 45.1 </td>
<td align="center"> 83.0 </td>
<td align="center"><a href="https://drive.google.com/file/d/1dqY3fts8caCxjZCiHhFJKCCQMkDUWaoW/view?usp=sharing">download</a></td>
</tr>
</tbody></table>


### ADE20K Panoptic Segmentation
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">PQ<sup>thing</sup></th>
<th valign="bottom">PQ<sup>stuff</sup></th>
<th valign="bottom">ckpt</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="./configs/ade20k/panoptic_segmentation/kmax_r50.yaml">ResNet-50</td>
<td align="center"> 42.2 </td>
<td align="center"> 81.6 </td>
<td align="center"> 50.4 </td>
<td align="center"> 41.9 </td>
<td align="center"> 42.7 </td>
<td align="center"><a href="https://drive.google.com/file/d/1ayqi5WyzHzVJPOr4odZ08Iz2Z7mqTEoy/view?usp=sharing">download</a></td>
 <tr><td align="left"><a href="./configs/ade20k/panoptic_segmentation/kmax_convnext_large.yaml">ConvNeXt-Large</a></td>
<td align="center"> 50.0 </td>
<td align="center"> 83.3 </td>
<td align="center"> 59.1 </td>
<td align="center"> 49.5 </td>
<td align="center"> 50.8 </td>
<td align="center"><a href="https://drive.google.com/file/d/12GQff3b4tozxGV2-L4wTUBKkmd7-aW5G/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

## Example Commands for Training and Testing
To train kMaX-DeepLab with ResNet-50 backbone:
```
python3 train_net.py --num-gpus 8 --num-machines 4 \
--machine-rank MACHINE_RANK --dist-url DIST_URL \
--config-file configs/coco/panoptic_segmentation/kmax_r50.yaml
```
The training takes 53 hours with 32 V100 on our end.

To test kMaX-DeepLab with ResNet-50 backbone and the provided weights:
```
python3 train_net.py --num-gpus NUM_GPUS \
--config-file configs/coco/panoptic_segmentation/kmax_r50.yaml \
--eval-only MODEL.WEIGHTS kmax_r50.pth
```

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/fun-research/kMaX-DeepLab)

## Porting TensorFlow Weights
We also provide a [script](./convert-tf-weights-to-d2.py) to convert the official TensorFlow weights into PyTorch format and use them in this code-base.

Example for porting and evaluating kMaX with ConvNeXt-Large on Cityscapes from [TensorFlow weights](https://github.com/google-research/deeplab2/blob/main/g3doc/projects/kmax_deeplab.md):
```
pip3 install tensorflow==2.9 keras==2.9
wget https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/kmax_convnext_large_res1281_ade20k_train.tar.gz
tar -xvf kmax_convnext_large_res1281_ade20k_train.tar.gz
python3 convert-tf-weights-to-d2.py ./kmax_convnext_large_res1281_ade20k_train/ckpt-100000 kmax_convnext_large_res1281_ade20k_train.pkl
python3 train_net.py --num-gpus 8 --config-file configs/ade20k/kmax_convnext_large.yaml \
--eval-only MODEL.WEIGHTS ./kmax_convnext_large_res1281_ade20k_train.pkl 
```

This expexts to give PQ = 50.6620. Note that minor performance difference may exist due to numeric difference across different deep learning frameworks and implementation details.


## Citing kMaX-DeepLab

If you find this code helpful in your research or wish to refer to the baseline
results, please use the following BibTeX entry.

*   kMaX-DeepLab:

```
@inproceedings{kmax_deeplab_2022,
  author={Qihang Yu and Huiyu Wang and Siyuan Qiao and Maxwell Collins and Yukun Zhu and Hartwig Adam and Alan Yuille and Liang-Chieh Chen},
  title={{k-means Mask Transformer}},
  booktitle={ECCV},
  year={2022}
}
```

*   CMT-DeepLab:

```
@inproceedings{cmt_deeplab_2022,
  author={Qihang Yu and Huiyu Wang and Dahun Kim and Siyuan Qiao and Maxwell Collins and Yukun Zhu and Hartwig Adam and Alan Yuille and Liang-Chieh Chen},
  title={CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation},
  booktitle={CVPR},
  year={2022}
}
```

## Acknowledgements
We express gratitude to the following open-source projects which this code-base is based on:

[DeepLab2](https://github.com/google-research/deeplab2)

[Mask2Former](https://github.com/facebookresearch/Mask2Former)