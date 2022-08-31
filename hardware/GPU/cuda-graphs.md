## 为了解决什么问题？
 较小的 kernel，其 CPU 开销相比较 GPU 更大的情况。

传统方法是给 CUDA 的 stream 里提交任务。而CUDA Graphs 是一种新的给 CUDA 提交任务的模型。图由一系列操作组成，比如 kernel luanch，以及依赖关系。依赖关系是独立于执行过程来定义的。这样图就可以一次定义，多次重复运行。而且定义和执行分开，就可以：

1. 跟 stream 方式对比，CPU的发射时间可以节省出来，因为大部分stepup 工作可以提前做
2. 给 CUDA 呈现出更多工作流之后，就可以做更多优化，而 stream 那种模式下是不行的

stream 模式下，当你把 kernel 放到 stream里，host 驱动需要执行一系列操作来准备 kernel 在GPU 上执行。这些操作在**每个 kernel 上都需要来一遍**

## 原理
一个 CUDA graph 是工作（主要是 kerne 及其参数）的记录，通过使用相同的参数来回放它。这样通过静态的图来换取极大程度降低 CPU 开销。图的参数和执行的kernel 都是固定的，所以图的回放能跳过参数设置和kernel dispatch，包括 python，C++，cuda driver 开销。底层是通过一个 cudaGraphLaunch 调用来让整个图进行提交。

![](./imgs/cuda-graphs-benefits.png)

通过图提交的工作，分为三个阶段：

1. 定义: 图里节点及之间的依赖关系
2. 初始化(Instantiation)：拿到图模版的快照，验证它，然后执行设置和初始化，目标是让下面的执行(launch) 过程能尽量少做事。产出是 **an executable graph**
3. 执行: 可以launch这个 executable graph到 stream 里。无论 luanch 多少次，都能避免上述的初始化(instantiation)过程

## 适用于哪些场景？效果如何？


## 有哪些限制？

上述原理可知，需要图是静态的：static shapes, static control flow

下面这些会造成 runtime error：

下面这些可能会造成数值错误或者未定义的错误

Dynamic control flow

Dynamic shapes: graph 假设每个被捕获的 op 里，每个 tensor 在每次重放时都是固定大小的

Make sure to run few warmup iterations before you capture, especially if you're setting torch.backends.cudnn.benchmark = True. You do NOT want all the cudnn benchmarking dry-run kernels ending up in the capture


[Prohibited and Unhandled Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#prohibited-unhandled-operations)

## 参考资料
[Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) ： 里面资料很丰富，在 Mask-R RCNN 上能加速1.7倍，nccl里的launch时间也能节省

[cuda graphs的一些限制](https://pytorch.org/docs/1.11/notes/cuda.html#constraints): 比如fwd和bwd的输入必须是静态固定在虚拟地址空间里的

[CUDA Graphs NV 介绍](https://developer.nvidia.com/blog/cuda-graphs/)

[CUDA Graphs Session](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32082/)
