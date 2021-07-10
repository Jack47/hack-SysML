## TODO
1. 看下 Gradient-Based Learning Applied to Document Recognition、 ImageNet、AlexNet 的论文
2. 自己实现一些常见的算子，这样了解的更加深刻。可以参考 lectuer 、智能计算系统等
3. 七月份目标是主攻卷积神经网络
4. 看下 dataloader 里 workers 之间的 multiprocessing 实现，图解 Pytorch DataLoader
6. Megatron LM
7. 看看 torch.autograd.backward 的实现
8. 可以看看 ActNN 里压缩算法在 GPU 上怎么写的，起始就是简单的加减乘除

## 问题
1. flops，显存，通信各自如何衡量出来？
2. data 时间如何计算的？

## 2021-7-10
2:20 pm- 

2h15m

11:20 am -1:35 pm
[Reading ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://arxiv.org/pdf/2104.14129.pdf)

## 2021-7-9

6:35 am - 7:10

Reading PyTorch Autograd docs
Reading MegaTron LM: [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/pdf/2104.04473.pdf)

## 2021-7-8
1h 45 min

10:15 pm -12:00 

## 2021-7-7
1h 40 min

7:20-8:00 am
fairscale checkpointing with test

9:40-10:40 pm

## 2021-7-6
1h 20min

10:10 - 11:00

7:30-8:00
[亚线性内存优化—activation checkpointing在oneflow中的实现](https://zhuanlan.zhihu.com/p/373662730)
[Explore Gradient Checkpointing in PyTorch](https://qywu.github.io/2019/05/22/explore-gradient-checkpointing.html)
结合 fairscale 看看它在显存优化方面的工作

[Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616)(2020-6-17) MegaEngine 实现了

## 2021-7-5
1h 10min

10:30pm - 11:20 pm

7:20-7:40
Paper: [ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf), [My Notes](imagenet-classification-with-deep-convolutional-neural-networks.md)

## 2021-7-4
4h20min

10:30 am - 12:00: train resnet101, batchsize 1k, epoch 100

13:00 - 13:20 
read pod, prototype DistModule, record questions

13:40- (-30) - 4:00 pm
read pytorch ddp source code: distributed.py

9:00 pm - 1:14 am 
draw DDP keynote, write blog

## 2021-7-3
6h

14:20 pm - 20:00 pm

动画详解 Pytorch Data Distributed Parallel

## 2021-7-2
2h

10:30 pm 12:00 pm

7:15 am - 7:40 am

1. [pytorch 里ddp 如何具体实现](https://pytorch.org/docs/stable/notes/ddp.html)？
2. [pytorch tunning performance guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
2. 如何画出动图：DDP

https://github.com/pytorch/pytorch/issues/37030

## 2021-7-1
2.5 h

10:00 pm - 12:09 pm
[Read pytorch ddp paper](./papers/papers/PyTorch Distributed-Data Parallel Training.md) and [draw illustration of ddp]()

7:30 - 8:00 am
[Read pytorch dataloder source code](./framworks/DataLoader.md)

## 2021-6-30
2 h

11:05- 12:00 pm
[multiprocessing in pytorch](./frameworks/multiprocessing.md)

6:40-7:50 am Read dataloader useage in prototype and pod

## 2021-6-29
2h 40min

6:40 - 7: 20 

10:20 - 12:30 ： [pytorch dataloader](./framworks/DataLoader.md)

## 2021-6-28
2h

6:10 - 7:20 am
Reading GShard

10:20 -  12:11 pm

[cs 294-ai-sys: lecture 1](./courses/ucberkely-cs294-ai-sys
l1.md)

## 2021-6-27
3.5h

7:55-8:30 am

8:30-9:00 pm

9:30 - 11:00 pm

1. Read GSPMD
2. Read GShard

## 2021-6-26
2h

8:30 pm -  9:30 pm

11:00 pm - 12:12

1. Read ['Zhanghuaizheng's Zhihu article](https://www.zhihu.com/question/315611053/answer/676815240)

2. Read CSE 599W: Systems for ML, Lecture1(Distributed Training and Communication Protocols), Lecture 5(GPU)
3. LaunchPad
