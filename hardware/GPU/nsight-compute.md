它支持收集很多指标，

`sm__inst_executed_pipe_tensor_op_hmma.sum` 代表 HMMA 指令执行了多少次

图里的例子：

Wavefronts: 对于每个request，在最终产生的唯一的“work package”，一个wavefront里的所有work item是并行处理的，但是不同wavefront里的work items是串行化并且在不同的cycles里执行的。每个request里，至少有一个 wavefront会被生成

Shared: 在每个chip上，所以相比local或者global memory，有更高的带宽，更低的延迟。它是在一个计算的 CTA 里共享的

这个 kernel 里，没用到 shared memory，而L2 Cache命中率42.32%，L1是36.91%

如何 apply rules(比如 Uncoalesced Memory Access，  non-fused floating-point instructions  ) 来找到

占用率，效率怎么算？

shared 和 cache （L1/L2）不是一回事


## 资料
1. [各种指标的含义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)
