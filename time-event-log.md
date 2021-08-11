## TODO
1. 看下 Gradient-Based Learning Applied to Document Recognition、 ImageNet、AlexNet 的论文
3. 七月份目标是主攻卷积神经网络
6. Megatron LM
7. 图解 Pipeline 里的气泡，1F1B 和 interleaved
8. 看 Megatron 第一篇论文，有助于理解 Transformer
9. 围绕着 oneflow, megatron 以及 ZeRO，结合源码(fairscale/layers) 把这些概念吃透。周末搞个 4D 的演示，interleaved pipeline
10. 看看 CheckMate: optimal tensor rematerialization (2019)
11. Superneurons: Dynamic GPU memory management for training deep neural networks. vdnn (2016)
12. 看 DeepSpeed 的实现

## 问题
1. flops，显存，通信各自如何衡量出来？
2. data 时间如何计算的？

## 2021-8-9
20 min
10:45 pm - 11:05 Reading Zero

## 2021-8-8
18 min

11:00 pm - 11:18 Finish Capuchin

## 2021-8-7
15 min

9:00 pm - 9:15 pm Flexible systems are the next frontier of machine learning
## 2021-8-6
11:00 pm  - 11:10 Find Codahale metrics implmemented in python

## 2021-8-5
9:00 - 9:30 Capuchin

## 2021-8-3
1h30min Reading [Capuchin: Tensor-based GPU Memory Management for Deep Learning](./papers/)
10:00 pm -11:00pm
9:00pm-9:30 pm

## 2021-8-2

9:15 pm - 10:22 pm successfully compiled actnn

## 2021-8-1
2h 30 min 

10:25 pm - 11:30 pm # install actnn in sh38

6:00 pm - 6:30 pm 
9:45 am - 11:00 am

1. actnn practice
2. characterizing 
## 2021-7-31
3h 30 min
10:25 pm - 10:55 pm
3:10 pm - 6:10 pm Reading [Characterizing Deep Learning Training Workloads on Alibaba-PAI](./papers/characterizing-deep-learning-training-workloads-on-alibaba-pai.md)

## 2021-7-30
20 min 
打印一下 PAI 的那篇论文，看一下杨老师发的一些资料

## 2021-7-29
30min
11:20 pm - 11:50 pm try install actnn

## 2021-7-28
20 min

10:40 pm - 11:00 看 Oneflow 群里的东西

## 2021-7-27
20 min
10:40 pm - 11:00

## 2021-7-26
40 min

7:20 am - 8:00 am. Read Rammer Paper

## 2021-7-25
4h 20min

9:00 pm - 11:50 pm
2:00 pm - 4:00 pm

10:30 am - 11:00 am

## 2021-7-24
2h30min
10:00 pm - 12:30 pm

8:00 pm - 9:00 pm Reading A Study of Persistent Thread Style GPU Programming for GPGPU Workloads

## 2021-7-23
50min
7:20 am - 8:10 am

## 2021-7-22
1h20min

10:00 pm - 11:20 pm

## 2021-7-21
8:45 pm - 9:45 pm. [Oneflow zhihu articles](./frameworks/oneflow/misc.m)
7:45 am - 9:00 am
model/transformer.py

## 2021-7-20

1h

samples in dataloader
10:30 pm - 11:00 pm
6:20 pm - 6:50 pm

## 2021-7-19
2h20min
5:20 pm  - 6:30
9:30 pm - 10:10
10:40 pm - 11:10 pm

## 2021-7-18

1h40min
4:50 - 6:10 pm draw 3d parallel graph, [megatron lm source code](./frameworks/megatron-lm/source-code.md)
8:10 - 30 am

## 2021-7-17
2:00 pm - 2:30
Find JIT usage, P2P communication in Megatron LM

## 2021-7-16
10:15 pm - 11:30

## 2021-7-15
1h

10:00 pm - 11:00

Read Efficient Large Scale Language Model Training, [my Notes](papers/efficient-large-scale-language-model-training.md)

## 2021-7-14
看下 Megatron 里的实现？结合oneflow 的文章，理解为啥难
或看懂 zero 及实现

## 2021-7-14
1h30min

10:30 pm - 12:00

## 2021-7-13
25 min

11:45pm- 12:10
Reading qscheme.py: compute\_quantization\_bits
layers.py: track\_running\_stats

## 2021-7-12
1h40min

10:00 pm - 11:40pm

Chat with friends about ML

Reading ActNN source code

## 2021-7-11

10:30 pm - 12:00 
Reading ActNN source code, train POD, review MR

## 2021-7-10
10h

9:40 pm - 12:00 pm

2:20 pm- 7:00 pm

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
