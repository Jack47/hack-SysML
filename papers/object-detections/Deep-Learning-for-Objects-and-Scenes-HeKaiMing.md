## Introduction

## Convolution Neural Networks: Recap
### LeNet

### Convolution
* Locally-connected
* Spatially weight-sharing: weight-sharing is a key in DL (e.g. RNN shares weights temporally)。通过这种共享权重，能极大减少模型的参数，而泛化能力不变
* Subsampling：干啥的呢？Pooling or stride to convolutions
* Fully-connected outputs: 最后这个 layer 有点像 SVM，是个线性分类器
* Train by BackProp

### AlexNet
在 LeNet 的backbond 基础上：

* ReLU：RevoLUtion of deep learning 因为 Accelerate training; better grad prop (vs. tanh) 这个 better 是指速度还是精度？
* Dropout : In-network ensembling, Reduce overfitting(might be instead done by BN)
* Data augmentation: Label-preserving transformation & Reduce overfitting.


### VGG

### GoogleNet

### Batch Norm

## ResNet
## ResNeXt


