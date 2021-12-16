<!-- vscode-markdown-toc -->
* 1. [Papers](#Papers)
	* 1.1. [Convolution Neural Networks](#ConvolutionNeuralNetworks)
	* 1.2. [Sparsity](#Sparsity)
	* 1.3. [Transformers](#Transformers)
	* 1.4. [Attribute Recognition](#AttributeRecognition)
	* 1.5. [Classification](#Classification)
	* 1.6. [Object Detection](#ObjectDetection)
	* 1.7. [Knowledge Distillation](#KnowledgeDistillation)
	* 1.8. [Network Communication](#NetworkCommunication)
		* 1.8.1. [Gradient Compression](#GradientCompression)
	* 1.9. [OPs in Network](#OPsinNetwork)
		* 1.9.1. [Batch Normalization](#BatchNormalization)
		* 1.9.2. [Group Normalization](#GroupNormalization)
	* 1.10. [Compute Efficiency](#ComputeEfficiency)
	* 1.11. [Memory Efficiency](#MemoryEfficiency)
	* 1.12. [Compression](#Compression)
	* 1.13. [Parallelism](#Parallelism)
		* 1.13.1. [Data Parallel](#DataParallel)
		* 1.13.2. [Pipeline Parallelism](#PipelineParallelism)
		* 1.13.3. [Parallelization Strategies](#ParallelizationStrategies)
	* 1.14. [Quantization](#Quantization)
	* 1.15. [Network](#Network)
	* 1.16. [Resource Management](#ResourceManagement)
	* 1.17. [Parameter Server](#ParameterServer)
* 2. [Network Operators](#NetworkOperators)
* 3. [Course：](#Course)
	* 3.1. [[CS179: GPU Programming](http://courses.cms.caltech.edu/cs179/)](#CS179:GPUProgramminghttp:courses.cms.caltech.educs179)
	* 3.2. [ [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)](#CS231n:ConvolutionalNeuralNetworksforVisualRecognitionhttp:cs231n.stanford.edu)
	* 3.3. [[CSE 599W: Systems for ML](http://dlsys.cs.washington.edu/schedule)](#CSE599W:SystemsforMLhttp:dlsys.cs.washington.eduschedule)
		* 3.3.1. [[Introduction to Deep Learning](http://dlsys.cs.washington.edu/pdf/lecture1.pdf)](#IntroductiontoDeepLearninghttp:dlsys.cs.washington.edupdflecture1.pdf)
		* 3.3.2. [[Lecture 1：Distributed Training and Communication Protocols](http://dlsys.cs.washington.edu/pdf/lecture11.pdf)](#Lecture1DistributedTrainingandCommunicationProtocolshttp:dlsys.cs.washington.edupdflecture11.pdf)
		* 3.3.3. [[Lecture 3: Overview of Deep Learning System](http://dlsys.cs.washington.edu/pdf/lecture3.pdf)](#Lecture3:OverviewofDeepLearningSystemhttp:dlsys.cs.washington.edupdflecture3.pdf)
		* 3.3.4. [[Lecture 5: GPU Programming](http://dlsys.cs.washington.edu/pdf/lecture5.pdf)](#Lecture5:GPUProgramminghttp:dlsys.cs.washington.edupdflecture5.pdf)
	* 3.4. [https://ucbrise.github.io/cs294-ai-sys-sp19/](#https:ucbrise.github.iocs294-ai-sys-sp19)
* 4. [Books：](#Books)
	* 4.1. [Deep Learning](#DeepLearning)
		* 4.1.1. [Backpropation](#Backpropation)
* 5. [OpenSource Frameworks And Libs](#OpenSourceFrameworksAndLibs)
	* 5.1. [TIMM](#TIMM)
	* 5.2. [[Tensor Comprehensions]() 2018.2.13 Facebook AI Research Technical Report](#TensorComprehensions2018.2.13FacebookAIResearchTechnicalReport)
	* 5.3. [TensorFlow RunTime (TFRT)](#TensorFlowRunTimeTFRT)
	* 5.4. [OneFlow](#OneFlow)
	* 5.5. [[Pytorch]()](#Pytorch)
		* 5.5.1. [JIT](#JIT)
	* 5.6. [[Tensorflow]()](#Tensorflow)
	* 5.7. [[Acme: A Research Framework for Distributed Reinforcement Learning](https://arxiv.org/pdf/2006.00979.pdf)(arXiv: June 1 2020)](#Acme:AResearchFrameworkforDistributedReinforcementLearninghttps:arxiv.orgpdf2006.00979.pdfarXiv:June12020)
	* 5.8. [[Launchpad]()](#Launchpad)
	* 5.9. [Reverb(2020)](#Reverb2020)
	* 5.10. [[Weld](https://www.weld.rs/)(Standford)](#Weldhttps:www.weld.rsStandford)
	* 5.11. [Table Data Storage](#TableDataStorage)
* 6. [Characterizing ML Training Workloads](#CharacterizingMLTrainingWorkloads)
* 7. [Data Preprocessing](#DataPreprocessing)
* 8. [Videos](#Videos)
* 9. [Hardware](#Hardware)
* 10. [ML Compilers](#MLCompilers)
	* 10.1. [Rammer](#Rammer)
	* 10.2. [IREE(Intermediate Representation Execution Environment)](#IREEIntermediateRepresentationExecutionEnvironment)
* 11. [Profiling](#Profiling)
	* 11.1. [Memory Profiling](#MemoryProfiling)
	* 11.2. [Communication Profiling](#CommunicationProfiling)
* 12. [Parallel Programming](#ParallelProgramming)
* 13. [Mixed Precision Training](#MixedPrecisionTraining)
	* 13.1. [CUDA](#CUDA)
* 14. [Visualizing Neural Networks](#VisualizingNeuralNetworks)
* 15. [Cluster Scheduler](#ClusterScheduler)
* 16. [Misc](#Misc)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

##  1. <a name='Papers'></a>Papers
ML Papers:

###  1.1. <a name='ConvolutionNeuralNetworks'></a>Convolution Neural Networks
[Gradient-Based Learning Applied to Document Recognition]()(1996?) : [Notes](./papers/LeNet.md), 在 OCR 领域里，用不需要像之前的方法一样，需要很多人工介入/设计/处理。对字母的大小，变体等有鲁棒性。更多依赖于从数据里自动学习，而非手工针对任务做的启发式设计。这样一个统一结构的模型可以代替过去好几个独自设计的模块

[ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)(2010) ：用 deep convolutional neural network 在 2010 年的 ImageNet LSVRC 比赛上分类 1200百万高清(当时的)图片。使用的网络有6000万参数和65万神经元，引入了 dropout 来防止过拟合。 [Notes](./papers/ImageNet.md)

[Going Deeper with Convolutions]() (ILSVRC 14) [Notes](./papers/Inception-GoogLeNet.md)

[Rich feature hierarchies for accurate object detection]() (2014) [Notes](./papers/object-detection/R-CNN.md)

Deformable Convolutional Networks (ICCV 2017) [Notes](./papers/deformable-convolutional-networks.md)

###  1.2. <a name='Sparsity'></a>Sparsity
[Accelerating Sparse Approximate Matrix Multiplication on GPUs](), [Notes](./papers/sparsity/Accelerating-Sparse-Approximate-Matrix-Multiplication-on-GPUs.md)

###  1.3. <a name='Transformers'></a>Transformers
[Attention is All you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Google Brain 2017) [My Notes](./papers/transformer/attention-is-all-you-need.md)

ViT: Vision Transformer: [An Image is worth 16x16 Words: Transformers for Image Recognition at Scale]() (2021.6.3) [My Notes](./papers/transformer/vit-an-image-is-worth-16x16-words.md)

lightseq: Accelerated Training for Transformer-based Models on GPUs(2021.10): 从工程角度出发实现 transformer 的训练和推理加速。[My Notes](./papers/transformer/lightseq.md) [源码阅读分析](./frameworks/lightseq/source-code.md)

Masked Autoencoders Are Scalable Vision Learners [My Notes](./papers/transformer/Masked-AutoEncoders-Are-Scalable Vision Learners.md)

###  1.4. <a name='AttributeRecognition'></a>Attribute Recognition
[Hierarchical Feature Embedding for Attribute Recognition]() (2020.5.23) [My Notes](./papers/attribute-recognition/hierarchical-feature-embedding-for-attribute-recognition.md)

###  1.5. <a name='Classification'></a>Classification



###  1.6. <a name='ObjectDetection'></a>Object Detection

[Fast R-CNN](): (2015) 提出用一个统一的网络来训练 R-CNN，而不是之前的三个阶段，不同的网络. [My Notes](./papers/object-detection/Fast-R-CNN.md)
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) (2016)：提出了 Region Proposal Network (RPN)，和检测网络共享图片卷积特征，因此 region proposal 的代价非常小。 [My Notes](./papers/object-detection/Faster-R-CNN.md)

[YOLO]() : (2016): [My Notes](./papers/object-detection/YOLO.md)

### Model Compression
Model compression(2006)：ensemble 模式下的多个模型的知识，可以压缩到一个模型里

蒸馏算是从上述模型压缩进一步发展而来的

###  1.7. <a name='KnowledgeDistillation'></a>Knowledge Distillation
[Contrastive Representation Distillation](https://arxiv.org/pdf/1910.10699.pdf) (2020) [My Notes](./papers/knownledge-distillation/contrastive-representation-distillation.md)

Search to Distill: Pearls are Everywhere but not the Eyes(2020) 除了参数外，还考虑了网络结构的蒸馏，结果发现好的老师无法教出在各方面都很优秀的学生  [My Notes](./papers/knownledge-distillation/search-to-distill.md)

A Gift from Knowledge Distillation: Fast optimization, Network Minimization and Transfer Learning (2017 韩国):  [My Notes](./papers/knowledge-distillation/a-gift-from-knowledge-distillation.md)

Distilling the Knowledge in a Neural Network Geoffrey Hinto 和 Jeff Dean(2015): 第一次提出模型蒸馏的概念及方法, 可以通过蒸馏让小模型能力提升很高（是语音场景）。提出了一种新的 ensemble 方法 [My Notes](./papers/knowledge-distillation/distilling-the-knowledge-in-a-neural-network.md)

###  1.8. <a name='NetworkCommunication'></a>Network Communication
[Flare: Flexible In-Network Allreduce](https://arxiv.org/pdf/2106.15565.pdf) 2021-6-29:  SwitchML 不灵活，他们设计了一个交换机：by using as a building block PsPIN, a RISC-V architecture implementing the sPIN programming model. 

In-Network Aggregation，[slides](https://www.usenix.org/system/files/nsdi21_slides_sapio.pdf), [Video](https://www.usenix.org/conference/nsdi21/presentation/sapio) [笔记](/network-communication/SwitchML.md) 18 年的

paper: [Scaling Distributed Machine Learning with In-Network Aggregation](https://www.usenix.org/system/files/nsdi21-sapio.pdf)

[NVIDIA SHARP](./network-communication/NVIDIA-Scalable-Hierarchical-Aggregation-and-Reduction-Protocol.md): 看起来跟上述开源方案的非常相似，对比见[这里](./network-communication/SwitchML.md#compare). 16 年就发论文了

[NetReduce: RDMA-Compatible In-Network Reduction for Distributed DNN Training Acceleration](https://arxiv.org/pdf/2009.09736.pdf) : 华为做的，需要硬件实现，好处是不需要修改网卡或交换机。

[MPI](./network-communication/MPI.md): 并行编程中的一种编程模型， Message Passing Interface

[PMI](./network-communication/PMI.md): 并行编程中的Process Management System

####  1.8.1. <a name='GradientCompression'></a>Gradient Compression

[GRACE: A Compressed Communication Framework for Distributed Machine Learning](https://sands.kaust.edu.sa/papers/grace.icdcs21.pdf) (2021) : s. We instantiate GRACE on TensorFlow and PyTorch, and implement 16 such methods.
Finally, we present a thorough quantitative evaluation with a variety of DNNs (convolutional and recurrent), datasets and system configurations. We show that the DNN architecture affects the relative performance among methods. Interestingly, depending on the underlying communication library and computational cost of compression / decompression, we demonstrate that some methods may be impractical. GRACE and the entire benchmarking suite are available as open-source.

[Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)(2017.12,2020-6-23 modify) :  In this paper, we find 99.9% of the gradient exchange in distributed SGD is redundant . 它就是在 GRACE 框架基础上开发的一类算法

[Code: GRACE: A Compressed Communication Framework for Distributed Machine Learning](https://github.com/sands-lab/grace):  is an unified framework for all sorts of compressed distributed training algorithms

###  1.9. <a name='OPsinNetwork'></a>OPs in Network

####  1.9.1. <a name='BatchNormalization'></a>Batch Normalization

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)(2015): 用了 BN 后学习率可以调大，初始化过程更鲁棒。也有 regularization 作用，可以一定程度代替 drop out。 faster training and high performance. [My Notes](./papers/batch-normalization.md)

Layer Normalization (2016.7) : [My Notes ](./papers/layer-normalization.md) : 想在 RNN 里引入类似 BN 的方法，来让训练速度提升，收敛的更快

[Norm matters: efficient and accurate normalization schemes in deep networks](https://arxiv.org/pdf/1803.01814.pdf)(2019 ): suggest several alternatives to the widely used L2 batch-norm, using normalization in L1 and L∞ space

####  1.9.2. <a name='GroupNormalization'></a>Group Normalization

Sys Papers:

###  1.10. <a name='ComputeEfficiency'></a>Compute Efficiency
[AdderNet: Do We Really Need Multiplications in Deep Learning](https://arxiv.org/abs/1912.13200v2)

[Rammer]() [My Notes](./papers/Rammer-Enabling-Holistic-DL-Optimizations-with-rTasks.md)

[1 Bit Adam]() () [My Notes](./network-communication/1-bit-Adam.md): 实现了 GPU 上的 error compensated Adam preconditioned momentum SGD。减少了 GPU 和 CPU间通信，还加速了优化器的计算

###  1.11. <a name='MemoryEfficiency'></a>Memory Efficiency

[MONeT: Memory optimization for deep networks](https://openreview.net/pdf?id=bnY0jm4l59)(ICLR 2021) : 发在了 Learning Representation 上，说明不是系统的改进，是算法的改进
1. MONeT: Memory Optimization for Deep Networks：https://github.com/utsaslab/MONeT

[Backprop with approximate activations for memory-efficient network training]() (arXiv 2019)

[Don't waste your bits! squeeze activations and gradients for deep neural networks via tinyscript]() (ICML, 2020)

上面这俩都是对所有元素采用同样的量化方法

Gist: Efficient data encoding for deep neural network training 2018

[ZeRO]()：大模型训练的标配，把 DDP 模式下冗余的显存都给分片了，需要时从别人那里要过来 [My Notes](./papers/ZeRO.md)

[ZeRO-offload]() 2021.1.18 [My Notes](./papers/ZeRO-offload.md) : 基于 ZeRO-2，把 NLP中 Adam 优化器实现在了 CPU 上，这样能 offload 优化器的内存和计算，这个是大头

[Capuchin: Tensor-based GPU Memory Management for Deep Learning]()(2020 ASPLOS) , [My notes](./papers/capuchin.md): 目标是为了节省内存，能训更大的 batchsize。内存管理到了 tensor 级别，而且是模型/计算图无关。在运行时 profile 出来是 swap 合适还是 recompute 合适。有 prefetch / evict 机制

Dynamic tensor rematerializatio(2020): 是 Checkpoint 核心是发明的这个线上算法不需要提前知道模型信息，就能实时产出一个特别好的 checkpointing scheme。这种在线方法能处理静态和动态图。算是陈天奇提供的方法的checkpoint 方法的后续：动态做，更优。[My Notes](./memory-efficiency/dynamic-tensor-rematerialization.md), [Source Code Notes](./frameworks/pytorch/dynamic-tensor-rematerialization.md)


Pushing deep learning beyond the gpu memory limit via smart swapping. (2020)

Tensor-based gpu memory management for deep learning. (2020)

[ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://arxiv.org/pdf/2104.14129.pdf) (2021-7-6) [My notes](./papers/ActNN.md), [source code notes](ActNN-source-code.md)

[Low-Memory Neural Network Training: A Technical Report](https://arxiv.org/pdf/1904.10631.pdf)(2019-4-24)

[Gradient/Activation Checkpointing](./memory-efficiency/gradient-checkpointing.md)
[Checkpoint in fairscale](./memory-efficiency/ckpt_activ_fairscale.md)
[Visual Gifs to show gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing)

Binaryconnect: Training deep neural networks with binary weights during propagations. (2015)

[ZeRO Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning]()(2021.4.16), [My Notes](./memory-efficiency/ZeRO-Infinity.md) : 在 ZeRO 基础上，把显存交换扩展到了 NVMe 硬盘上。ZeRO 系列的好处是不需要改用户的代码。ZeRO 系列主要解决内存问题（Memory wall)

PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management: 实现基于 Transformer 的 NLP 里预训练场景下，高效的 swap 实现机制，让模型训练更高效，能够最大程度复用不会同时存在的 chunks [My Notes](./memory-efficiency/patrickstar.md)， [Source Code Notes](./frameworks/patrick-star.md)
###  1.12. <a name='Compression'></a>Compression

Deep compression: Compressing deep nerual networks with pruning, trained quantization and huffman coding. (ICLR, 2016 Song Han)

Deep gradient compression: Reducing the communication bandwidth for distributed training. (ICLR, 2018 Songhan)

###  1.13. <a name='Parallelism'></a>Parallelism

[Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/pdf/2104.04473.pdf) (2021.4): 主要介绍了继 Megatron-LM 之后，如何结合多种并行方式，让特定 batchsize 的大transformer 模型，高通吐地运行起来。[阅读笔记](papers/efficient-large-scale-language-model-training.md)

####  1.13.1. <a name='DataParallel'></a>Data Parallel
[PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/pdf/2006.15704.pdf)(2020.6.28) [My notes](./papers/PyTorch Distributed-Data Parallel Training.md)

[Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training](https://arxiv.org/abs/2004.13336)(2020-4-28) : 提出了 weights 的自动切分方法，通过高效的通信原语来同步，使用静态分析计算图的方法，应用于 ADAM 或 SGD

[GSPMD: General and Scalable Parallelization for ML Graphs](https://arxiv.org/pdf/2105.04663.pdf)(2021-5-10) (for transformers)



####  1.13.2. <a name='PipelineParallelism'></a>Pipeline Parallelism
[PipeDream: Generalized Pipeline Parallelism for DNN Training](https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf)(SOSP'19)

[GPipe: Efficient training of giant neural networks using pipeline parallelism]()(NIPS 2019)

[PipeDream Source Code](https://github.com/msr-fiddle/pipedream)

[fairscale pipeline parallelism source code](https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/pipe)

[torochpipe](https://github.com/kakaobrain/torchgpipe)

####  1.13.3. <a name='ParallelizationStrategies'></a>Parallelization Strategies
[Beyond Data and Model Parallelism for Deep Neural Networks](https://arxiv.org/pdf/1807.05358.pdf)
> Defined a more comprehensive search space of parallelization strategies for DNNs called SOAP, which includes strategies to parallelize a DNN in the Sample, Operation, Attribute, and Parameter dimesions. Proposed FlexFlow, a deep learning framework that uses guided randomized search of the SOAP spaceto find a fast parallelization strategy for a specific parallel machine. To accelerate this search, FlexFlow introduces a novel execution simulator that can accurately predict a parallelizaiton strategy's performance.

###  1.14. <a name='Quantization'></a>Quantization 

And the bit goes down: Revisiting the quantization of neural networks. (ICLR 2020)

MQBench Towards Reproducible and Deployable Model Quantization Benchmark (NIPS 2021 workshop) 提出了一套验证量化算法在推理上可复现性和可部署性的Benchmark集合，指出了当前学术界和工业界之间的 Gap。[My Notes](./papers/quantization/MQBench.md)

Google. Gemmlowp: building a quantization paradigm from first principles 里面提到了量化感知训练的方法


###  1.15. <a name='Network'></a>Network
[GradientFlow: Optimizing Network Performance for Distributed DNN Training on GPU Clusters](https://arxiv.org/pdf/1902.06855.pdf)(cs.DC 2019)

> Proposed a communication backend named GradientFlow for distributed DNN training. First we integrate ring-based all-reduce, mixed-precision training, and computation/communication overlap. Second, we propose lazy allreduce to improve network throughput by fusing multiple communication operations into a singe one, and design coarse-grained sparse communication to reduce network traffic by transmitting important gradient chunks.

###  1.16. <a name='ResourceManagement'></a>Resource Management

[Optimus on top of K8s](https://i.cs.hku.hk/~cwu/papers/yhpeng-eurosys18.pdf)

###  1.17. <a name='ParameterServer'></a>Parameter Server
[Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)(2013?)

### NLP

Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers [Notes](,/papers/NLP/scale-efficiently_insights-from-pretraining-and-fine-tuning-transformers.md)

### Recommenders 

PERSIA: An Open, Hybrid System Scaling Deep Learning-based Recommenders up to 100 Trillion Parameters(2021) [Notes](./papers/recommender/persia.md)

##  2. <a name='NetworkOperators'></a>Network Operators
[Dropout vs Droppath](operators/dropout.md)

[Conv](operators/conv.md)
##  3. <a name='Course'></a>Course：
[CS294-AI-Sys](https://ucbrise.github.io/cs294-ai-sys-sp19/)(UC Berkely)

[笔记](./courses/ucberkely-cs294-ai-sys/)


###  3.1. <a name='CS179:GPUProgramminghttp:courses.cms.caltech.educs179'></a>[CS179: GPU Programming](http://courses.cms.caltech.edu/cs179/)
[Week 4 Lec 10: cuBLAS Intro](http://courses.cms.caltech.edu/cs179/2021_lectures/cs179_2021_lec10.pdf). [My Notes](./courses/cs179/cuBLAS.md)

里面提到的一些经典论文，找到人讨论效果更佳


###  3.2. <a name='CS231n:ConvolutionalNeuralNetworksforVisualRecognitionhttp:cs231n.stanford.edu'></a> [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
>  This course is a deep dive into the details of deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. During the 10-week course, students will learn to implement and train their own neural networks and gain a detailed understanding of cutting-edge research in computer vision. Additionally, the final assignment will give them the opportunity to train and apply multi-million parameter networks on real-world vision problems of their choice. Through multiple hands-on assignments and the final course project, students will acquire the toolset for setting up deep learning tasks and practical engineering tricks for training and fine-tuning deep neural networks.
> 

(Vector, Matrix, and Tensor Derivatives(Erik Learned-Miller))[http://cs231n.stanford.edu/vecDerivs.pdf] [My Notes](./courses/cs231n/vector-derivatives.md)

###  3.3. <a name='CSE599W:SystemsforMLhttp:dlsys.cs.washington.eduschedule'></a>[CSE 599W: Systems for ML](http://dlsys.cs.washington.edu/schedule)
> System aspect of deep learning: faster training, efficient serving, lower
memory consumption

有不少结合TVM的部分，比如 Memory Optimization, Parallel Scheduing, 其中我感兴趣的部分：

####  3.3.1. <a name='IntroductiontoDeepLearninghttp:dlsys.cs.washington.edupdflecture1.pdf'></a>[Introduction to Deep Learning](http://dlsys.cs.washington.edu/pdf/lecture1.pdf)
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

####  3.3.2. <a name='Lecture1DistributedTrainingandCommunicationProtocolshttp:dlsys.cs.washington.edupdflecture11.pdf'></a>[Lecture 1：Distributed Training and Communication Protocols](http://dlsys.cs.washington.edu/pdf/lecture11.pdf)

####  3.3.3. <a name='Lecture3:OverviewofDeepLearningSystemhttp:dlsys.cs.washington.edupdflecture3.pdf'></a>[Lecture 3: Overview of Deep Learning System](http://dlsys.cs.washington.edu/pdf/lecture3.pdf)
Computational Graph Optimization and Execution

Runtime Parallel Scheduing / Networks


####  3.3.4. <a name='Lecture5:GPUProgramminghttp:dlsys.cs.washington.edupdflecture5.pdf'></a>[Lecture 5: GPU Programming](http://dlsys.cs.washington.edu/pdf/lecture5.pdf)

###  3.4. <a name='https:ucbrise.github.iocs294-ai-sys-sp19'></a>https://ucbrise.github.io/cs294-ai-sys-sp19/
[AI-Sys Spring 2019](https://ucbrise.github.io/cs294-ai-sys-sp19/)(UC Berkeley)

##  4. <a name='Books'></a>Books：
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)

###  4.1. <a name='DeepLearning'></a>Deep Learning
[Deep Learning](https://www.deeplearningbook.org/): Ian Goodfellow and Yoshua Bengio and Aron Courville
> intended to help students and practitioners enter the field of machine learning in general and deep learning in particular. The online version of the book is now complete and will remain available online for free.

####  4.1.1. <a name='Backpropation'></a>Backpropation
[How the backpropagation algorithms works](./books/neural-networks-and-deep-learning/backpropagation.md)

[Customize backward function in pytorch](./frameworks/pytorch/functions.md)

##  5. <a name='OpenSourceFrameworksAndLibs'></a>OpenSource Frameworks And Libs

###  5.1. <a name='TIMM'></a>TIMM
PyTorch Image Models (TIMM) is a library for state-of-the-art image classification. With this library you can:
* Choose from 300+ pre-trained state-of-art image classification models.
* Train models afresh on research datasets such as ImageNet using provided scripts
* Finetune pre-trained modes on your own datasets, including the latest cutting edge models.

Pytorch Image Models

###  5.2. <a name='TensorComprehensions2018.2.13FacebookAIResearchTechnicalReport'></a>[Tensor Comprehensions]() 2018.2.13 Facebook AI Research Technical Report

[My notes](./papers/frameworks/tensor-comprehensions.md)

###  5.3. <a name='TensorFlowRunTimeTFRT'></a>TensorFlow RunTime (TFRT)

> aims to provide a unified, extensble infrastructure layer with best-in-class performance across a wide variety of domain specific hardware.

[TFRT DeepDive](https://drive.google.com/drive/folders/1fkLJuVP-tIk4GENBu2AgemF3oXYGr2PB)

目前还在比较早期阶段，还无法做到通用

###  5.4. <a name='OneFlow'></a>OneFlow

###  5.5. <a name='Pytorch'></a>[Pytorch]()
 [Functions/APIs in PyTorch ](./frameworks/pytorch/functions.md)
 
 [Reading Source Code Snippets](./frameworks/pytorch/source-code.md)
 
####  5.5.1. <a name='JIT'></a>JIT

![](https://github.com/pytorch/tvm/blob/master/pt_execution.png?raw=true)

###  5.6. <a name='Tensorflow'></a>[Tensorflow]()
[Tensorflow: a system for large-scale machine learning]()(OSDI 2016)

[Automatic differetiation in Pytorch]()(2017)

[Mesh-tensorflow: Deep learning for supercomputers]()(NIPS 2018)

###  5.7. <a name='Acme:AResearchFrameworkforDistributedReinforcementLearninghttps:arxiv.orgpdf2006.00979.pdfarXiv:June12020'></a>[Acme: A Research Framework for Distributed Reinforcement Learning](https://arxiv.org/pdf/2006.00979.pdf)(arXiv: June 1 2020)
> Agents are different scales of both complexity and computation -- including distributed versions.

让 RL 从单机版原型可以无痛扩展到多机分布式。
###  5.8. <a name='Launchpad'></a>[Launchpad]()

From Acme Paper:
> Roughly speaking, Launchpad provides a mechanism for creating a distributed program as a graph cnosisting of nodes and edges. Nodes exactly correspond to the modules -- represented as class instances as described above -- whereas the edges represent a client/server channel allowing communication between two modules. The key innovation of Launchpad is that it handles the creating of these edges in such a way that from perspective of any module there is no ditinction between a local and remote communication, e.g. for an actor retrieving parameters from a learner in both instances this just looks like a method call.

直观感觉是这些领域里需要一些基础的框架和库，去解决分布式通信，把问题抽象，和具体场景解耦出来，这样研究人员复用这些轮子。跟当年互联网领域 rpc 通信框架、最近几年的微服务里服务中心、服务主键等功能类似。

![](./imgs/acme-fig4-distrbuted-asynchronous-agent.png)

###  5.9. <a name='Reverb2020'></a>Reverb(2020)
The dataset in RL can be backed by a low-level data storaeg system.
> It enables efficient insertion and routing of items and a flexible sampling mechanism that allows:FIFO, LIFO, unifrom, and weighted sampling schemes.


###  5.10. <a name='Weldhttps:www.weld.rsStandford'></a>[Weld](https://www.weld.rs/)(Standford)
Fast rust parallel code generation for data analytics frameworks. Developed at Standford University.
> Weld is a language and runtime for improving the performance of data-intensive applications. It optimizes across libraries and functions by expressing the core computations in libraries using a common intermediate repersentation, and optimizing across each framwork.
> Modern analytics applications combine multiple functions from different libraries and frameworks to build complex workflows.
> Weld's take on solving this problem is to lazily build up a computation for the entire workflow, and then optimizingn and evaluating it only when a result is needed.
 
看起来就是解决掉用不同的库和框架来构建复杂工作流时的效率问题。通过引入一层中间表达，然后再实现到不同的框架里来做联合优化。

Paper: [Weld: Rethinking the Interface Between Data-Intensive Applications.](https://arxiv.org/abs/1709.06416)
> To address this problem, we propose Weld, a new interface between data-intensive libraries that can optimize across disjoint libraries and functions. Weld can be integrated into existing frameworks such as Spark, TensorFlow, Pandas and NumPy without chaning their user-facing APIs.

Related research papers: [PipeDream: General Pipeline Parallelism for DNN Training](https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf) (SOSP'19)

###  5.11. <a name='TableDataStorage'></a>Table Data Storage
[Reverb is an efficient and easy-to-use data storage and trasport system designed for machine learning research](https://github.com/deepmind/reverb)(DeepMind, 2021-4-22)

> Used an an experience replay system for distributed reinforcement learning algorithms

##  6. <a name='CharacterizingMLTrainingWorkloads'></a>Characterizing ML Training Workloads
[Characterizing Deep Learning Training Workloads on Alibaba-PAI](https://arxiv.org/pdf/1910.05930)(2019-10-14): 主要看看他们是用什么方法来profile的，是否有借鉴意义

##  7. <a name='DataPreprocessing'></a>Data Preprocessing
[ImageNet-21K Pretraining for the Masses]() (2021.8.5) [My Notes](./papers/data/imagenet-21k-pretraining-for-the-masses.md)

##  8. <a name='Videos'></a>Videos
[Flexible systems are the next frontier of machine learning](https://www.youtube.com/watch?v=Jnunp-EymJQ&list=WL&index=14) Stanford 邀请 Jeff Dean 和 Chris Re 探讨最新的进展(2019) [My Notes](./videos/Flexible-systems-are-the-next-frontiner-of-machine-learning.md)

##  9. <a name='Hardware'></a>Hardware

### GPU

[NVLink](./hardware/GPU/nvlink.md) vs PCiE

[Mixed Precision Training](https://arxiv.org/abs/1710.03740) (2017 ICLR) [My Notes](./papers/mixed-precision-training.md)

[APEX: NVIDIA mixed-precision training](./hardware/GPU/apex.md)

[What Every Programmer Should Know About Floating-Point Arithmetic](https://floating-point-gui.de/) [My Notes](./hardware/GPU/what-every-programmer-should-know-about-floating-point.md)

[fp16](./hardware/GPU/fp16.md)

### CUDA
CUTLASS: [My Notes](./hardware/GPU/cutlass.md)

cuBLAS: [My Notes](./hardware/GPU/cublas.md)

[pytorch cuda api](./hardware/GPU/pytorch-cuda.md)

CUDA API: [My Notes](./hardware/GPU/CUDA-API.md)

[Cooperative Groups](./hardware/GPU/cooperative_groups.md)

[CUB: Provides state of the art, reusable software components for every layer of the CUDA programming model](./hardware/GPU/cub.md)

[SoftMax 的高效 CUDA 实现](./hardware/GPU/efficient-softmax.md)

[CUDA 里的全局 threadId 如何计算？](./hardware/GPU/indexes-in-cuda.md)

[CUDA Graphs: 可以用来节省传统 stream 方式下 cudaLaunchKernel 的时间，适合小的静态 kernel](./hardware/GPU/cuda-graphs.md)

Optimizing Convolutitonal Layers: NV 官网的优化卷积运算的指南。[Notes](./hardware/GPU/optimizing-conv-layers.md)
##  10. <a name='MLCompilers'></a>ML Compilers
###  10.1. <a name='Rammer'></a>Rammer

###  10.2. <a name='IREEIntermediateRepresentationExecutionEnvironment'></a>IREE(Intermediate Representation Execution Environment)
> It's an MLIR-based end-to-end compiler and runtime that lowers ML modeles to a unified IR that scales up to meet the needs of the datacenter and down to satisfy the constraints and special considerations of mobile and edge deployments.
目前支持 TensorFlow, JAX

##  11. <a name='Profiling'></a>Profiling
###  11.1. <a name='MemoryProfiling'></a>Memory Profiling
[Capuchin: Tensor-based GPU Memory Management for Deep Learning]()(2020 ASPLOS) , [My notes](./papers/capuchin.md)

###  11.2. <a name='CommunicationProfiling'></a>Communication Profiling
[Designing-a-profiling-and-visualization-tool-for-scalable-and-in-depth-analysis-of-high-performance-gpu-clusters]()(2019 HiPC), [My notes](./papers/designing-a-profiling-and-visualization-tool-for-scalable-and-in-depth-analysis-of-high-performance-gpu-clusters.md)

##  12. <a name='ParallelProgramming'></a>Parallel Programming
[Persistent threads style gpu programming for gpgpu workloads]() (2012 IEEE) [My Notes](papers/persistent-threads.md)
[Persistent Threads Block](https://www.classes.cs.uchicago.edu/archive/2016/winter/32001-1/papers/AStudyofPersistentThreadsStyleGPUProgrammingforGPGPUWorkloads.pdf)



##  14. <a name='VisualizingNeuralNetworks'></a>Visualizing Neural Networks
[Deep Learning in your browser](https://cs.stanford.edu/people/karpathy/convnetjs/)

##  15. <a name='ClusterScheduler'></a>Cluster Scheduler
[Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized-Deep-Learning]() [My notes](papers/scheduler/pollux.md)

AntMan

##  16. <a name='Misc'></a>Misc
[The Modern Histry of Object Recognition -- Infographic](https://medium.com/@nikasa1889/the-modern-history-of-object-recognition-infographic-aea18517c318)

[HPC-oriented Latency Numbers Every Programmer Should Know](https://gist.github.com/understeer/4d8ea07c18752989f6989deeb769b778)

[Curriculum Learning for Natural Language Understanding](https://aclanthology.org/2020.acl-main.542.pdf)()

[On The Power of Curriculum Learning in Training Deep Networks](https://arxiv.org/abs/1904.03626) (2019) : 传统的要求batch 的训练数据是均匀分布，而这个是模拟了人的学习过程：从简单的任务开始，逐步增加难度。解决了两个问题：1）根据训练难度排序数据集 2）计算逐步增加了难度的 一系列 mini-batches

[What every computer scientist should know about floating-point arithmetric](https://dl.acm.org/doi/10.1145/103162.10316)
