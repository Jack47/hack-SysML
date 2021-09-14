1. 在解决的是什么问题？显存发展不够，大模型缺显存而跑不起来的问题。
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？不需要做模型拆分，不需要修改研究员的代码。提出了 bandwidth centric 的分片方式，这样可以利用多机的带宽，而非集中到一台主机上。memory-centirc tiling (这个应该是为了不拆分模型而做的）
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
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

## 问题

## 启发
1. 有个关于 DL 中并行的 survey：Ben-Nun and Hoefler[19]: Demystifying parallel and distributed deep learning: An in-depth concurrency analysis. 2019
