## Introduction

## Convolution Neural Networks: Recap

### LeNet

#### Convolution
* Locally-connected
* Spatially weight-sharing: weight-sharing is a key in DL (e.g. RNN shares weights temporally)。通过这种共享权重，能极大减少模型的参数，而泛化能力不变
* Subsampling：干啥的呢？Pooling or stride to convolutions
* Fully-connected outputs: 最后这个 layer 有点像 SVM，是个线性分类器
* Train by BackProp

### AlexNet
在 LeNet 的backbond 基础上：

* ReLU：RevoLUtion of deep learning 因为 Accelerate training; better grad prop (vs. tanh) 这个 better 是指速度还是精度？ 指速度，能更快学习到
* Dropout : In-network ensembling, Reduce overfitting(might be instead done by BN)
* Data augmentation: Label-preserving transformation & Reduce overfitting.


### VGG
#### Modularized Design
* 3x3 Conv as the module
* Stack the same module
* Same computation for each module(1/2 spatial size => 2x filters)

#### Stage-wise Training：啥意思
* VGG-11 => VGG-13 => VGG-16
* We need a better initialization
### Initialization
Xavier/MSRA init

### GoogleNet
Xavier/MSRA init 无法直接应用到多分枝的网络里，比如 GoogleNet

Very Good accuracy, with small footprint（咋理解？）. 里面有很多非常好的设计：

* Multiple branches: 比如 1x1，3x3，5x5 和 Pool
* Shortcuts: stand-alone 1x1, merged by concat。 所以不需要 dimension decrease
* Bottleneck: Reduce dim by 1x1 before expensive 3x3/5x5 conv. 不掉点么？
 
 后来的各种变体，依然保留了上述三个特性
 
此时 BN 大显身手

### Batch Norm

Normalizing image input 在 1998 年 LeCun 提出 Efficient Backprop 时就有了

Xavier/MSRA init：Analytic normalizing each layer

BN: data-driven normalizing each layer, for each mini-batch:

* 极大加速训练
* 对初始化不敏感了
* 提高了 Regularization



## ResNet
此时如果只是 stacking layers，会发现效果并不好：training error 和 test error 都反而变高了
## ResNeXt


