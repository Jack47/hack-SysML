## TODO
看看微软官网的几篇文章介绍，有动图

1. 在解决的是什么问题？显存发展不够，大模型缺显存而跑不起来的问题。
2. 为何成功，标志/准是什么？ 自动做分片的
3. 在前人基础上的关键创新是什么？不需要做模型拆分，不需要修改研究员的代码。提出了 bandwidth centric 的分片方式，这样可以利用多机的带宽，而非集中到一台主机上。memory-centirc tiling (这个应该是为了不拆分模型而做的）
4. 关键结果有哪些？能支撑未来1000倍的大模型，到 1000T。
5. 有哪些局限性？如何优化？ 没法解决模型算力太大的问题
6. 这个工作可能有什么深远的影响？


## 1. 扩展介绍

大模型在大量范型数据集上进行预训练（pre-train），然后把这个模型进行fine-tune，这样可以用上游的模型在下游不同应用使用。fine-tune 代价更低，需要的GPU 时长显著缩短，甚至在一台主机8卡上也能做。

1. Infinify offload engine
2. Memory-centric tiling

主要五大贡献：

1. LM 训练中内存和性能特点（sec 3） 以及 带宽需求
2. ZeRO Infinity: a) offload engine: 同时使用 GPU, CPU 和 NVMe 内存，以及 GPU 和 CPU 算力 b) memory-centric tiling，可以处理巨大的单个 OP，而不需要模型并行 c) bandwidth centric 分片，利用所有并行设备上的聚合后内存带宽 d) overlap centric design: 让计算和通信重叠 e) ease-inspired implementation 来避免模型代码的改动
3. 大量评估实验，说明：训练效率，扩展性等
4. Infinity 对硬件设计的潜在影响: sec 9
5. 开源版本的实现

## 2. 相关工作
ZeRO：移除了 DDP 中的内存冗余，方法是把三类数据(优化器状态，梯度，参数)进行分片。在 ZeRO-3 里，每个层的参数是由唯一的数据并行的进程所拥有的。通过做 broadcast 通信来让 forward/backward时一个 op 需要的参数在op执行前准备好。这样ZeRO-3保证每个数据并行的进程只更新它自己拥有的参数上的优化器状态.

ZeRO Offload: 有哪些局限性？如何优化？是针对 NLP 里 模型状态和优化器占用内存最大，针对混合精度训练和 Adam 优化器设计的。而且参数依然需要存储在GPU 里一份，而且不能用 ZeRO-3（不然 cpu->gpu->网络，开销较大）。所以在大 batchsize 下效果才好。后来 ZeRO-Infinity 克服了这些局限

Adam 在大模型训练里常用，以占用更多内存的方式，给模型参数和梯度维护了一阶和二阶统计数据

bandwidth-centric partitioning + powerful communication overlap-centric design + high performance NVMe access 

## 显存需求
论文里主要基于 Transformer 架构的 billion 级别模型 +  Adam 来分析的内存需求

Model states: optimizer states, gradients, model parameters。而这些都是跟参数有关，是 20*M。而 M = 12\*nl\*hd^2 => 240\*nl\*hd^2 。其中 nl: number of transformer layers，hd: hidden dimension

Residual states: primarily referring to activation memory，它取决于模型结构，batch size 和 sequence length. 可以通过 checkpointing 等技术显著减小占用。 ci: number of Transformer blocks between two activation checkpoints. bsz * seq * hd is the size of the input to each Transformer block.

GPU working memory: 为了支持训练，在 GPU 上需要有的最小量的显存大小，假设其他状态都可以 offload 出去

Figure 2 a: 展示了模型参数逐步变大时，Model states、每个节点上的 Activation，以及 MS 和 A 的 working memory 变化情况。


## 带宽需求

定义了 efficiency 的公式

主要是 Fig 3，这个图是计算出来的，还是实际测试出来的？图跟 AIT ，peak 有关。答：实际测试出来的，不同的 bs，固定的 sequence length: 1024. 其中有一些是根据公式计算的，比如 efficiency. 

Optimizer States 对带宽要求比较高：跟 parameter 和 梯度相比，需要4倍带宽。这个原因是它比他们大4倍把？

Bandwidth w.r.t activation memory:

这个反而是比较小的

## 5. 设计概览
### 5.1 给未来/前所未有的规模设计
5.1.1 Infinity offload engine for model states: Infinity 基于 ZeRO-3，把所有模型状态都分片。跟其他 ZeRO 系列不同，Infinity 有强大的 offload 机制，把所有分片的模型状态可以 offload 到 CPU 或者 NVMe 内存里，或者放到 GPU里，基于对内存的需求。

5.1.2 CPU Offload for activations: 除了上面提到的 Model states，Infinity 还可以 offload 激活值内存，如果有需要。

通过上述两个方法，基本上 **几百T** 的模型参数，也是能训练的

5.1.3 Memory-centric  tiling  for working memory : 把一个算子拆成了多个分片(tiles) ，顺序执行

### 5.2 Design for Excellent Training Efficiency
Offload 所有 model states 和 激活值到 CPU 或 NVMe，只有当带宽非常高效，才能达到预期目的。这个非常有挑战：CPU 内存带宽比 GPU 显存慢数量级，而 NVMe 的带宽又比 CPU 的慢数量级。

接下来讨论如何让 Infinity 达到上述的所需带宽：

![](./imgs/v100-dgx-2-bandwidth.png)

1: u

6.1.1

## 启发
1. 可以把 AllReduce 关闭，这样可以测试无网络情况下，各台主机速度有多块

## 问题
1. 是不是满足一些条件，才能发挥出威力？
2. 为什么能达到线性加速？原因是 NVMe 足够大？比 GPU，CPU 的内存都大。不像 ZeRO-2，里面 Parameters 是GPU 上也要存储一份，所以有上限
3. 没太看明白 Figure 2 b 里的东西：NVMe 的带宽比
4. 为啥 4.1， 4.2 里都提到 Activation checkpoints?
5. 4.2 里提到的 Bandwidth w.r.t Parameter and Gradients. 没理解是说实验表明？70GB/s 的带宽，就可以有 50% 的效率，即使是最小的 batch size？此时数据移动理论上就可以完全被计算掩盖？ 而我们用的是 PCie 3/4， 是 64GB/s

## 启发
1. 有个关于 DL 中并行的 survey：Ben-Nun and Hoefler[19]: Demystifying parallel and distributed deep learning: An in-depth concurrency analysis. 2019
2. 里面有 Transformer 如何计算内存占用的公式
