本文主要是帮助理解 GPU 基本的执行模式，方便理解特定的 Layers 或者神经网络在使用 GPU 时的效率

主要包含四部分

* 基本的 GPU 架构
* 算子如何划分并并发执行的（GPU 执行模式）
* 使用**计算密集程度(arithmetic intensity)**预估算术密集性算子的性能瓶颈
* DL 中算子的大概划分种类和对应的性能瓶颈

比如 A100，每个 GPU 有 108个 SMs，一个 40MB 的 L2 cache，高达 2039GB/s 的带宽，从 80G 的 HBM2 DRAM 里取数据

![](./imgs/simplified-gpu-arch.png)

## GPU Execution Model

两个非常重要的概念：

1. GPU 使用两级的thread层次来执行线程。给定的线程会被划分为等量大小的thread blocks，一堆 thread blocks 会被发射来执行 kernel 函数. Tiles Quantizations, Waving。所以一般blocks 的数量得是SM的4倍以上，而每个 thread block 里有几百个线程，那差不多有足够的并行度
2. GPU 掩盖互相依赖的指令延迟的方法是切换到其他线程的执行上。因此，高效利用 GPU 的线程数量需要比核数要多。一个 SM 上可以并发执行多个 thread blocks

## 三种性能瓶颈因子

在给定处理器上的性能限制主要是以下三个因素：内存带宽，算术带宽和延迟(只要上文提到的并发度够了，就还行)

## DNN 算子分三大类

### Elementwise Operations
可能是单目或者双目算子。关键是这类算子执行的算术操作，是在每个元素上执行的，不依赖于tensor 里的其他元素。比如 ReLU(x) = max(0,x)。这类一般会是 memory-limited

### Reduction Operations
在一组输入 tensor 值上面进行计算，然后产出结果。

例如，(avg)pooling layers， BN，SoftMax。通常是 memory limited

### Dot-Product Operations
两个输入 tensor 的点乘，通常需要一个权重 tensor 和一个激活值 tensor

本文来自：https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
