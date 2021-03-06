
实现细粒度的模型并行。

![](https://pic4.zhimg.com/80/v2-8e342553ff46dd88d16e265ed530b5b7_720w.jpg)

主要贡献是提供了一套原语，允许高层开发者通过简单的方式指导代码生成（编译期）生成对应的模型并行的代码。

## 摘要
GShard 由一套轻量级注解 API 组成，是 XLA 编译器的扩展。它提供优雅的方式来以对现有模型最小的改动去表达大量并行计算的模式。GShard 让我们可以用自动sharding的方式，把Sparsely-Gated Mixture-of-Experts 里的多语言神经机器翻译 Transformer 模型扩攒到超过6k亿参

## 1. 介绍
### 1.1 Scaling遇到的挑战
训练超大规模模型时，当容量超过了当个加速器的内存，遇到如下的挑战：

1. 特定架构模型并行的支持：TensorFlow 和PyTorch缺乏高效的模型并行算法。简单地通过图划分来做模型并行是支持的，但是导致利用率很低，因为网络有顺序依赖，基于梯度的优化。为了扩展已有模型，用户通常需要投入很大的工程工作，例如把代码迁移到特定框架(Mesh-TensorFlow).

2. 计算代价和模型大小上的超级线性扩展：最直观增加模型大小的方法是增加深度和宽度，通常会导致最后训练时间线性增大。模型并行是通过划分层权重和计算到多个设备上，导致通信开销和设备低使用率。设备低使用率来源于分配不均衡和线性依赖。这种超线性无法分通过使用更多设备来解决。
3. 大模型表示上的基础设施扩展性：图在超大规模模型分布在上千个设备上的表示会成为 DL 框架和优化器的瓶颈。例如，通过op间划分来增加更多D倍的层，或通过op内部划分到D个设备上增加模型维度会导致图里有O（D）个节点。设备间的通信通道会把图的大小增加到 O(D^2)(例如分片的聚合或转置）。这类增加图大小的方法导致超大规模模型里图的构建和编译时间不可解。
4. 实现分片策略代价较高：把一个模型高效分片到多个设备上很有挑战，因为需要设备间对应的通信。对于图级别的划分，需要复杂算法来减少分配在不同设备上不同部分之间的线性依赖性。对于操作符级别的并行，不同划分方式有不同的通信模式，取决于语义，比如，是否需要累积部分结果，或需要重新排列数据分片。而且 TensorFlow 里有大量 ad-hoc 语义的算子，导致更加复杂。

### 1.2 规模化高效训练的设计原则


## 7 结论
条件计算效果非常好，在模型大小和计算代价上取得了折衷。

有一个恰当的抽象层可以 把模型描述从并行视线中分离开，让模型开发者关注在网络实现上，让 GShard自动划分计算图，产出在所有设备上并行运行的程序。产出单个的足够通用的程序来表达在所有并行设备上的计算是编译可扩展的关键。传统的给不同分片产出多个特定程序的方法会导致扩展到上千个分区时遇到编译时间爆炸的问题。为了解决这个复杂度，我们基于 SPMD 分片，这种允许Tensor任意维度可以被划分的方法，引入了多种编译革新。

