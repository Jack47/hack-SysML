来自：https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/

一句话总结： RAPIDS 组里研发的 RMM 系列比 CUDA 里默认的 cudaMalloc 和 cudaFree 性能要好。主要能力在加速，而非节省显存上。

它的特点：
* 提供了可以定制分配 device 和 host 内存的机制
* 定义了接口
* 一堆数据结构来支持上述接口（有一些默认的实现）

## RMM 里的子分配(suballocation)
为什么能达到高性能？通过从底层拿到一块大的内存，然后切分到更小的块给上面的应用层使用。大部分高效的分配器都是这样工作的。

几个默认的实现：
1. *pool_memory_resource* : 实现了 coalescing 的 suballocator，当释放时从周边的邻居结合成更大的块。

2. *binning_memory_resource* : 

里面有性能对比的图

## Resource adaptors: Logging and replay, limiting and tracking

