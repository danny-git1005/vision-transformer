# Vision Transformer

This is a practice on vision transformer( ViT ).
It is based on the paper "[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)"

# Architecture
![image](https://user-images.githubusercontent.com/63143667/225532733-a77220d1-2be3-4203-a6a7-bf15cd714a9b.png)
```
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
├─patch_embedding: 1-1                        [-1, 50, 1024]            --
|    └─Sequential: 2-1                        [-1, 49, 1024]            --
|    |    └─Rearrange: 3-1                    [-1, 49, 3072]            --
|    |    └─LayerNorm: 3-2                    [-1, 49, 3072]            6,144
|    |    └─Linear: 3-3                       [-1, 49, 1024]            3,146,752
|    |    └─LayerNorm: 3-4                    [-1, 49, 1024]            2,048
|    └─Dropout: 2-2                           [-1, 50, 1024]            --
├─TransformerEncoder: 1-2                     [-1, 50, 1024]            --
├─Sequential: 1-3                             [-1, 500]                 --
|    └─LayerNorm: 2-3                         [-1, 1024]                2,048
|    └─Linear: 2-4                            [-1, 500]                 512,500
===============================================================================================
Total params: 3,669,492
Trainable params: 3,669,492
Non-trainable params: 0
Total mult-adds (M): 111.16
===============================================================================================
Input size (MB): 0.57
Forward/backward pass size (MB): 4.27
Params size (MB): 14.00
Estimated Total Size (MB): 18.84
```
# Data
Image data is from kaggle "BIRDS 500 SPECIES- IMAGE CLASSIFICATION"(https://www.kaggle.com/datasets/gpiosenka/100-bird-species)
## File structure
```
birds
  ├── test
  │   ├── ABBOTTS BABBLER
  |   ├── ABBOTTS BOOBY
  |   |         .
  |   |         .
  |   |         .
  |
  ├── train
  │   ├── ABBOTTS BABBLER
  │   ├── ABBOTTS BABBLER
  |   |         .
  |   |         .
  |   |         .
  |
  ├── val
  │   ├── ABBOTTS BABBLER
  │   ├── ABBOTTS BABBLER
  |   |         .
  |   |         .
  |   |         .
```

Inference:\n
data:\n
https://www.kaggle.com/code/jainamshah17/pytorch-starter-image-classification
https://www.kaggle.com/code/lonnieqin/bird-classification-with-pytorch
https://www.kaggle.com/code/stpeteishii/bird-species-classify-torch-conv2d
https://www.kaggle.com/datasets/gpiosenka/100-bird-species/code?datasetId=534640&searchQuery=torch

ViT:\n
https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
https://medium.com/ai-blog-tw/detr%E7%9A%84%E5%A4%A9%E9%A6%AC%E8%A1%8C%E7%A9%BA-%E7%94%A8transformer%E8%B5%B0%E5%87%BAobject-detection%E6%96%B0pipeline-a039f69a6d5d
https://juejin.cn/post/6924173141409267726
https://www.cnblogs.com/wxkang/p/16150868.html
https://arxiv.org/pdf/2106.01548.pdf

tensoeboard:\n
https://github.com/HyoungsungKim/Pytorch-tensorboard_tutorial





