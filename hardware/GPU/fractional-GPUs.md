1. 在解决的是什么问题？实现一套在一个 GPU 上虚拟出多个 GPU，让应用之间计算和显存彼此隔离的方案
2. 为何成功，标志/准是什么？比 NV 自己的 MPS 实现还高效，而且提供显存的隔离
3. 在前人基础上的关键创新是什么？把 L2 cache 和 DRAM 都可以隔离（利用 page coloring）
4. 关键结果有哪些？比Nvidia 的 MPS 还要高效
5. 有哪些局限性？如何优化？ 需要修改 kernel 函数的实现
6. 这个工作可能有什么深远的影响？在云上需要这种强隔离的技术

## 摘要：
在实时场景下，希望在GPU上同时跑好几个GPU任务，同时让性能是可预期的。Nvidia 开始支持 MPS，它是以闭源形式的 kernel 模块来提供多个程序跑在GPU上的能力。但它并不能让这多个应用程序之间在共享的内存架构里互不影响。实现发现单纯隔离计算还不够，还需要考虑内存的隔离，否则由于多个任务的并发执行，相互之间可能会打断，读写事务可能会被拖慢到10倍以下

通过 micro-benchmark 发现，GPU的内存架构和CPU的非常不同，更适合使用 paging coloring 技术。基于我们的发现，可以实现把L2 cache 和 DRAM 划分给多个 GPU任务。这样提供了非常强的隔离性

## 介绍

CPU上即使两个程序运行在不同的核上，依然会影响对方，原因在于内存架构层级上的冲突（主要是共享的 cache 和 DRAM），原因是以下几个：

1. 共享的cache里，cache集合
2. Miss Status Holding Registers 在共享 cache下的冲突
3. 在 memory controller 上的重排序
4. DRAM 里的行缓存冲突
5. DRAM 总线竞争

有论文专门针对 CPU 下的上述冲突做了很多研究，设计机制来解决上述的竞争问题

本文主要贡献：

1. 实现了基于软件的 GPU 分片方案，提供高度的隔离
2. 使用 micro-benchmark 逆向了 GPU里的 L2 cache 和 DRAM
3. 首先在 GPU 上实现了 page coloring
4. 移植 Caffe 到 FGPU 的抽象上

## 2. 背景

![](./imgs/simplified-gpu-arch.png)

### A. 计算层次
GPU 里有硬件调度器，决定 blocks 和硬件 SM 之间的映射关系，一个 SM 执行多个 blocks。但由于一个 SM 一次只能执行固定个数的 blocks，所以一些 blocks 会被放到队列里，等待资源 ok

NV 并没有公开硬件调度的行为
### B. 内存层次
有三级： L1 cache, L2 cache, DRAM。L1是被所属的 SM 独占的，在 SM内的线程里共享。而所有的 SM 共享 L2和 DRAM。NV 没有公开GPU内存细节。所以下面介绍下 CPU 里的细节：

chip/bank/row

## 3. 计算隔离

实现：目前NV GPU没有提供硬件支持来把指定 SM 分配给一个 kernel。所以使用软件分区来达到计算隔离。基于 [32]来做的，persistent block
## 问题


[代码在这里](https://github.com/sakjain92/Fractional-GPUs)

[32]: Enabling and exploiting flexible task assignment on gpu through sm-centric program transformations(2015 Acm)
