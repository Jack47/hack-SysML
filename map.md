## Papers
ML Papers:

### Convolution Neural Networks
[Gradient-Based Learning Applied to Document Recognition]()(1996?) : [Notes](./papers/LeNet.md), 在 OCR 领域里，用不需要像之前的方法一样，需要很多人工介入/设计/处理。对字母的大小，变体等有鲁棒性。更多依赖于从数据里自动学习，而非手工针对任务做的启发式设计。这样一个统一结构的模型可以代替过去好几个独自设计的模块

[ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)(2010) ：用 deep convolutional neural network 在 2010 年的 ImageNet LSVRC 比赛上分类 1200百万高清(当时的)图片。使用的网络有6000万参数和65万神经元，引入了 dropout 来防止过拟合。 [Notes](./papers/ImageNet.md)

[Going Deeper with Convolutions]() (ILSVRC 14) [Notes](./papers/Inception-GoogLeNet.md)

[Rich feature hierarchies for accurate object detection]() (2014) [Notes](./papers/object-detection/R-CNN.md)

### Sparsity
[Accelerating Sparse Approximate Matrix Multiplication on GPUs](), [Notes](./papers/sparsity/Accelerating-Sparse-Approximate-Matrix-Multiplication-on-GPUs.md)

### Transformers
[Attention is All you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Google Brain 2017) [My Notes](./papers/transformer/attention-is-all-you-need.md)

ViT: Vision Transformer: [An Image is worth 16x16 Words: Transformers for Image Recognition at Scale]() (2021.6.3) [My Notes](./papers/transformer/vit-an-image-is-worth-16x16-words.md)

### Attribute Recognition
[Hierarchical Feature Embedding for Attribute Recognition]() (2020.5.23) [My Notes](./papers/attribute-recognition/hierarchical-feature-embedding-for-attribute-recognition.md)

### Classification



### Object Detection

[Fast R-CNN](): (2015) 提出用一个统一的网络来训练 R-CNN，而不是之前的三个阶段，不同的网络. [My Notes](./papers/object-detection/Fast-R-CNN.md)
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) (2016)：提出了 Region Proposal Network (RPN)，和检测网络共享图片卷积特征，因此 region proposal 的代价非常小。 [My Notes](./papers/object-detection/Faster-R-CNN.md)

[YOLO]() : (2016): [My Notes](./papers/object-detection/YOLO.md)

### Knowledge Distillation
[Contrastive Representation Distillation](https://arxiv.org/pdf/1910.10699.pdf) (2020) [My Notes](./papers/knownledge-distillation/contrastive)

### Network Communication
[Flare: Flexible In-Network Allreduce](https://arxiv.org/pdf/2106.15565.pdf) 2021-6-29:  SwitchML 不灵活，他们设计了一个交换机：by using as a building block PsPIN, a RISC-V architecture implementing the sPIN programming model. 

In-Network Aggregation，[slides](https://www.usenix.org/system/files/nsdi21_slides_sapio.pdf), [Video](https://www.usenix.org/conference/nsdi21/presentation/sapio) [笔记](/network-communication/SwitchML.md) 18 年的

paper: [Scaling Distributed Machine Learning with In-Network Aggregation](https://www.usenix.org/system/files/nsdi21-sapio.pdf)

[NVIDIA SHARP](./network-communication/NVIDIA-Scalable-Hierarchical-Aggregation-and-Reduction-Protocol.md): 看起来跟上述开源方案的非常相似，对比见[这里](./network-communication/SwitchML.md#compare). 16 年就发论文了

[NetReduce: RDMA-Compatible In-Network Reduction for Distributed DNN Training Acceleration](https://arxiv.org/pdf/2009.09736.pdf) : 华为做的，需要硬件实现，好处是不需要修改网卡或交换机。

[MPI](./network-communication/MPI.md): 并行编程中的一种编程模型， Message Passing Interface

[PMI](./network-communication/PMI.md): 并行编程中的Process Management System

#### Gradient Compression

[GRACE: A Compressed Communication Framework for Distributed Machine Learning](https://sands.kaust.edu.sa/papers/grace.icdcs21.pdf) (2021) : s. We instantiate GRACE on TensorFlow and PyTorch, and implement 16 such methods.
Finally, we present a thorough quantitative evaluation with a variety of DNNs (convolutional and recurrent), datasets and system configurations. We show that the DNN architecture affects the relative performance among methods. Interestingly, depending on the underlying communication library and computational cost of compression / decompression, we demonstrate that some methods may be impractical. GRACE and the entire benchmarking suite are available as open-source.

[Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)(2017.12,2020-6-23 modify) :  In this paper, we find 99.9% of the gradient exchange in distributed SGD is redundant . 它就是在 GRACE 框架基础上开发的一类算法

[Code: GRACE: A Compressed Communication Framework for Distributed Machine Learning](https://github.com/sands-lab/grace):  is an unified framework for all sorts of compressed distributed training algorithms

### OPs in Network

#### Batch Normalization

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)(2015): 用了 BN 后学习率可以调大，初始化过程更鲁棒。也有 regularization 作用，可以一定程度代替 drop out。 faster training and high performance 

Layer Normalization (2016.7) : [My Notes ](./papers/layer-normalization) : 想在 RNN 里引入类似 BN 的方法，来让训练速度提升，收敛的更快

[Norm matters: efficient and accurate normalization schemes in deep networks](https://arxiv.org/pdf/1803.01814.pdf)(2019 ): suggest several alternatives to the widely used L2 batch-norm, using normalization in L1 and L∞ space

#### Group Normalization

Sys Papers:

### Compute Efficiency
[AdderNet: Do We Really Need Multiplications in Deep Learning](https://arxiv.org/abs/1912.13200v2)

[Rammer]() [My Notes](./papers/Rammer-Enabling-Holistic-DL-Optimizations-with-rTasks.md)

[1 Bit Adam]() () [My Notes](./network-communication/1-bit-Adam.md): 实现了 GPU 上的 error compensated Adam preconditioned momentum SGD。减少了 GPU 和 CPU间通信，还加速了优化器的计算

### Memory Efficiency

[MONeT: Memory optimization for deep networks](https://openreview.net/pdf?id=bnY0jm4l59)(ICLR 2021) : 发在了 Learning Representation 上，说明不是系统的改进，是算法的改进
1. MONeT: Memory Optimization for Deep Networks：https://github.com/utsaslab/MONeT

[Backprop with approximate activations for memory-efficient network training]() (arXiv 2019)

[Don't waste your bits! squeeze activations and gradients for deep neural networks via tinyscript]() (ICML, 2020)

上面这俩都是对所有元素采用同样的量化方法

Gist: Efficient data encoding for deep neural network training 2018

[ZeRO]() [My Notes](./papers/ZeRO.md)

[ZeRO-offload]() 2021.1.18 [My Notes](./papers/ZeRO-offload.md) : 基于 ZeRO-2，把 NLP中 Adam 优化器实现在了 CPU 上，这样能 offload 优化器的内存和计算

[Capuchin: Tensor-based GPU Memory Management for Deep Learning]()(2020 ASPLOS) , [My notes](./papers/capuchin.md): 目标是为了节省内存，能训更大的 batchsize。内存管理到了 tensor 级别，而且是模型/计算图无关。在运行时 profile 出来是 swap 合适还是 recompute 合适。有 prefetch / evict 机制

Dynamic tensor rematerializatio(2020): [My Notes](./memory-efficiency/dynamic-tensor-rematerialization.md): 实现了根据前一个 iteration，来自动选择 。 TODO：看下论文

Pushing deep learning beyond the gpu memory limit via smart swapping. (2020)

Tensor-based gpu memory management for deep learning. (2020)

[ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://arxiv.org/pdf/2104.14129.pdf) (2021-7-6) [My notes](./papers/ActNN.md), [source code notes](ActNN-source-code.md)

[Low-Memory Neural Network Training: A Technical Report](https://arxiv.org/pdf/1904.10631.pdf)(2019-4-24)

[Gradient/Activation Checkpointing](./memory-efficiency/gradient-checkpointing.md)
[Checkpoint in fairscale](./memory-efficiency/ckpt_activ_fairscale.md)
[Visual Gifs to show gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing)

Binaryconnect: Training deep neural networks with binary weights during propagations. (2015)

[ZeRO Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning]()(2021.4.16), [My Notes](./memory-efficiency/ZeRO-Infinity.md) : 在 ZeRO 基础上，把显存交换到 CPU、NVMe 上。ZeRO 系列的好处是不需要改用户的代码。ZeRO 系列主要解决内存问题（Memory wall)

### Compression

Deep compression: Compressing deep nerual networks with pruning, trained quantization and huffman coding. (ICLR, 2016 Song Han)

Deep gradient compression: Reducing the communication bandwidth for distributed training. (ICLR, 2018 Songhan)

### Parallelism

[Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/pdf/2104.04473.pdf) (2021.4): 主要介绍了继 Megatron-LM 之后，如何结合多种并行方式，让特定 batchsize 的大transformer 模型，高通吐地运行起来。[阅读笔记](papers/efficient-large-scale-language-model-training.md)

#### Data Parallel
[PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/pdf/2006.15704.pdf)(2020.6.28) [My notes](./papers/PyTorch Distributed-Data Parallel Training.md)

[Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training](https://arxiv.org/abs/2004.13336)(2020-4-28) : 提出了 weights 的自动切分方法，通过高效的通信原语来同步，使用静态分析计算图的方法，应用于 ADAM 或 SGD

[GSPMD: General and Scalable Parallelization for ML Graphs](https://arxiv.org/pdf/2105.04663.pdf)(2021-5-10) (for transformers)



#### Pipeline Parallelism
[PipeDream: Generalized Pipeline Parallelism for DNN Training](https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf)(SOSP'19)

[GPipe: Efficient training of giant neural networks using pipeline parallelism]()(NIPS 2019)

[PipeDream Source Code](https://github.com/msr-fiddle/pipedream)

[fairscale pipeline parallelism source code](https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/pipe)

[torochpipe](https://github.com/kakaobrain/torchgpipe)

#### Parallelization Strategies
[Beyond Data and Model Parallelism for Deep Neural Networks](https://arxiv.org/pdf/1807.05358.pdf)
> Defined a more comprehensive search space of parallelization strategies for DNNs called SOAP, which includes strategies to parallelize a DNN in the Sample, Operation, Attribute, and Parameter dimesions. Proposed FlexFlow, a deep learning framework that uses guided randomized search of the SOAP spaceto find a fast parallelization strategy for a specific parallel machine. To accelerate this search, FlexFlow introduces a novel execution simulator that can accurately predict a parallelizaiton strategy's performance.

### Quantization 

And the bit goes down: Revisiting the quantization of neural networks. (ICLR 2020)


### Network
[GradientFlow: Optimizing Network Performance for Distributed DNN Training on GPU Clusters](https://arxiv.org/pdf/1902.06855.pdf)(cs.DC 2019)

> Proposed a communication backend named GradientFlow for distributed DNN training. First we integrate ring-based all-reduce, mixed-precision training, and computation/communication overlap. Second, we propose lazy allreduce to improve network throughput by fusing multiple communication operations into a singe one, and design coarse-grained sparse communication to reduce network traffic by transmitting important gradient chunks.

### Resource Management

[Optimus on top of K8s](https://i.cs.hku.hk/~cwu/papers/yhpeng-eurosys18.pdf)

### Parameter Server
[Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)(2013?)

## Network Operators
[Dropout vs Droppath](operators/dropout.md)
[Conv](operators/conv.md)
## Course：
[CS294-AI-Sys](https://ucbrise.github.io/cs294-ai-sys-sp19/)(UC Berkely)

[笔记](./courses/ucberkely-cs294-ai-sys/)


### [CS179: GPU Programming](http://courses.cms.caltech.edu/cs179/)
[Week 4 Lec 10: cuBLAS Intro](http://courses.cms.caltech.edu/cs179/2021_lectures/cs179_2021_lec10.pdf). [My Notes](./courses/cs179/cuBLAS.md)

里面提到的一些经典论文，找到人讨论效果更佳


###  [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
>  This course is a deep dive into the details of deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. During the 10-week course, students will learn to implement and train their own neural networks and gain a detailed understanding of cutting-edge research in computer vision. Additionally, the final assignment will give them the opportunity to train and apply multi-million parameter networks on real-world vision problems of their choice. Through multiple hands-on assignments and the final course project, students will acquire the toolset for setting up deep learning tasks and practical engineering tricks for training and fine-tuning deep neural networks.
> 

### [CSE 599W: Systems for ML](http://dlsys.cs.washington.edu/schedule)
> System aspect of deep learning: faster training, efficient serving, lower
memory consumption

有不少结合TVM的部分，比如 Memory Optimization, Parallel Scheduing, 其中我感兴趣的部分：

#### [Introduction to Deep Learning](http://dlsys.cs.washington.edu/pdf/lecture1.pdf)
##### Evolution of ConvNets:

LeNet(1998)
Alexnet(2012)
GoogleLeNet(2014): Multi-indepent pass way (Sparse weight matrix)

Inception BN (2015): Batch normalization

Residual Net(2015): Residual pass way

Convolution = Spatial Locality + Sharing

ReLU: y = max(x, 0). 

Why ReLU?

* Cheap to compute
* It is roughly linear

Dropout Regularization: Randomly zero out neurons with probability 0.5

Why Dropout? Overfitting prevention.

Batch Normalization: Stabilize the Magnitude.

* Subtract mean
* Divide by standard deviation
* Output is invariant to input scale!
  - Scale input by a constant
  - Output of BN remains the same


Impact:
* Easy to tune learning rate . ?
* Less sensitive initialization. ?

#### [Lecture 1：Distributed Training and Communication Protocols](http://dlsys.cs.washington.edu/pdf/lecture11.pdf)

#### [Lecture 3: Overview of Deep Learning System](http://dlsys.cs.washington.edu/pdf/lecture3.pdf)
Computational Graph Optimization and Execution

Runtime Parallel Scheduing / Networks


#### [Lecture 5: GPU Programming](http://dlsys.cs.washington.edu/pdf/lecture5.pdf)

### https://ucbrise.github.io/cs294-ai-sys-sp19/
[AI-Sys Spring 2019](https://ucbrise.github.io/cs294-ai-sys-sp19/)(UC Berkeley)

## Books：
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)

### Deep Learning
[Deep Learning](https://www.deeplearningbook.org/): Ian Goodfellow and Yoshua Bengio and Aron Courville
> intended to help students and practitioners enter the field of machine learning in general and deep learning in particular. The online version of the book is now complete and will remain available online for free.

## OpenSource Frameworks And Libs

### TIMM
PyTorch Image Models (TIMM) is a library for state-of-the-art image classification. With this library you can:
* Choose from 300+ pre-trained state-of-art image classification models.
* Train models afresh on research datasets such as ImageNet using provided scripts
* Finetune pre-trained modes on your own datasets, including the latest cutting edge models.

Pytorch Image Models

### [Tensor Comprehensions]() 2018.2.13 Facebook AI Research Technical Report

[My notes](./papers/frameworks/tensor-comprehensions.md)

### TensorFlow RunTime (TFRT)

> aims to provide a unified, extensble infrastructure layer with best-in-class performance across a wide variety of domain specific hardware.

[TFRT DeepDive](https://drive.google.com/drive/folders/1fkLJuVP-tIk4GENBu2AgemF3oXYGr2PB)

目前还在比较早期阶段，还无法做到通用

### OneFlow

### [Pytorch]()
#### JIT

![](https://github.com/pytorch/tvm/blob/master/pt_execution.png?raw=true)

### [Tensorflow]()
[Tensorflow: a system for large-scale machine learning]()(OSDI 2016)

[Automatic differetiation in Pytorch]()(2017)

[Mesh-tensorflow: Deep learning for supercomputers]()(NIPS 2018)

### [Acme: A Research Framework for Distributed Reinforcement Learning](https://arxiv.org/pdf/2006.00979.pdf)(arXiv: June 1 2020)
> Agents are different scales of both complexity and computation -- including distributed versions.

让 RL 从单机版原型可以无痛扩展到多机分布式。
### [Launchpad]()

From Acme Paper:
> Roughly speaking, Launchpad provides a mechanism for creating a distributed program as a graph cnosisting of nodes and edges. Nodes exactly correspond to the modules -- represented as class instances as described above -- whereas the edges represent a client/server channel allowing communication between two modules. The key innovation of Launchpad is that it handles the creating of these edges in such a way that from perspective of any module there is no ditinction between a local and remote communication, e.g. for an actor retrieving parameters from a learner in both instances this just looks like a method call.

直观感觉是这些领域里需要一些基础的框架和库，去解决分布式通信，把问题抽象，和具体场景解耦出来，这样研究人员复用这些轮子。跟当年互联网领域 rpc 通信框架、最近几年的微服务里服务中心、服务主键等功能类似。

![](./imgs/acme-fig4-distrbuted-asynchronous-agent.png)

### Reverb(2020)
The dataset in RL can be backed by a low-level data storaeg system.
> It enables efficient insertion and routing of items and a flexible sampling mechanism that allows:FIFO, LIFO, unifrom, and weighted sampling schemes.


### [Weld](https://www.weld.rs/)(Standford)
Fast rust parallel code generation for data analytics frameworks. Developed at Standford University.
> Weld is a language and runtime for improving the performance of data-intensive applications. It optimizes across libraries and functions by expressing the core computations in libraries using a common intermediate repersentation, and optimizing across each framwork.
> Modern analytics applications combine multiple functions from different libraries and frameworks to build complex workflows.
> Weld's take on solving this problem is to lazily build up a computation for the entire workflow, and then optimizingn and evaluating it only when a result is needed.
 
看起来就是解决掉用不同的库和框架来构建复杂工作流时的效率问题。通过引入一层中间表达，然后再实现到不同的框架里来做联合优化。

Paper: [Weld: Rethinking the Interface Between Data-Intensive Applications.](https://arxiv.org/abs/1709.06416)
> To address this problem, we propose Weld, a new interface between data-intensive libraries that can optimize across disjoint libraries and functions. Weld can be integrated into existing frameworks such as Spark, TensorFlow, Pandas and NumPy without chaning their user-facing APIs.

Related research papers: [PipeDream: General Pipeline Parallelism for DNN Training](https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf) (SOSP'19)

### Table Data Storage
[Reverb is an efficient and easy-to-use data storage and trasport system designed for machine learning research](https://github.com/deepmind/reverb)(DeepMind, 2021-4-22)

> Used an an experience replay system for distributed reinforcement learning algorithms

## Characterizing ML Training Workloads
[Characterizing Deep Learning Training Workloads on Alibaba-PAI](https://arxiv.org/pdf/1910.05930)(2019-10-14): 主要看看他们是用什么方法来profile的，是否有借鉴意义

## Data Preprocessing
[ImageNet-21K Pretraining for the Masses]() (2021.8.5) [My Notes](./papers/data/imagenet-21k-pretraining-for-the-masses.md)

## Videos
[Flexible systems are the next frontier of machine learning](https://www.youtube.com/watch?v=Jnunp-EymJQ&list=WL&index=14) Stanford 邀请 Jeff Dean 和 Chris Re 探讨最新的进展(2019) [My Notes](./videos/Flexible-systems-are-the-next-frontiner-of-machine-learning.md)

## Hardware
[NVLink](./hardware/GPU/nvlink.md) vs PCiE
## ML Compilers
### Rammer

### IREE(Intermediate Representation Execution Environment)
> It's an MLIR-based end-to-end compiler and runtime that lowers ML modeles to a unified IR that scales up to meet the needs of the datacenter and down to satisfy the constraints and special considerations of mobile and edge deployments.
目前支持 TensorFlow, JAX

## Profiling
### Memory Profiling
[Capuchin: Tensor-based GPU Memory Management for Deep Learning]()(2020 ASPLOS) , [My notes](./papers/capuchin.md)

### Communication Profiling
[Designing-a-profiling-and-visualization-tool-for-scalable-and-in-depth-analysis-of-high-performance-gpu-clusters]()(2019 HiPC), [My notes](./papers/designing-a-profiling-and-visualization-tool-for-scalable-and-in-depth-analysis-of-high-performance-gpu-clusters.md)

## Parallel Programming
[Persistent threads style gpu programming for gpgpu workloads]() (2012 IEEE) [My Notes](papers/persistent-threads.md)
[Persistent Threads Block](https://www.classes.cs.uchicago.edu/archive/2016/winter/32001-1/papers/AStudyofPersistentThreadsStyleGPUProgrammingforGPGPUWorkloads.pdf)

## Mixed Precision Training
[https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740) (2017 ICLR)

### CUDA
cuBLAS:

[pytorch cuda api](./hardware/GPU/pytorch-cuda.md)

CUDA API: [My Notes](./hardware/GPU/CUDA-API.md)

[CUB: Provides state of the art, reusable software components for every layer of the CUDA programming model](./hardware/GPU/cub.md)

## Visualizing Neural Networks
[Deep Learning in your browser](https://cs.stanford.edu/people/karpathy/convnetjs/)

## Cluster Scheduler
[Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized-Deep-Learning]() [My notes](papers/scheduler/pollux.md)

AntMan

## Misc
[The Modern Histry of Object Recognition -- Infographic](https://medium.com/@nikasa1889/the-modern-history-of-object-recognition-infographic-aea18517c318)

[HPC-oriented Latency Numbers Every Programmer Should Know](https://gist.github.com/understeer/4d8ea07c18752989f6989deeb769b778)

[Curriculum Learning for Natural Language Understanding](https://aclanthology.org/2020.acl-main.542.pdf)()

[On The Power of Curriculum Learning in Training Deep Networks](https://arxiv.org/abs/1904.03626) (2019) : 传统的要求batch 的训练数据是均匀分布，而这个是模拟了人的学习过程：从简单的任务开始，逐步增加难度。解决了两个问题：1）根据训练难度排序数据集 2）计算逐步增加了难度的 一系列 mini-batches

[What every computer scientist should know about floating-point arithmetric](https://dl.acm.org/doi/10.1145/103162.10316)
