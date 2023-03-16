# vision-transfprmer

This is a practice on vision transformer( ViT ).
It is based on the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"(https://arxiv.org/pdf/2010.11929.pdf)

The above is the archtecture : 
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
Image data is from kaggle "BIRDS 500 SPECIES- IMAGE CLASSIFICATION"(https://www.kaggle.com/datasets/gpiosenka/100-bird-species)
