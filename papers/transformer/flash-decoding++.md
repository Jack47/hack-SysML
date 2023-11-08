## 3 Asynchronized Softmax with Unified Maximum Value
把类似 flash attention 计算过程中的同步 softmax 改为异步，可以解决 20% 开销：容许不同部分的 softmax 独立计算，无须同步更新。因为发现多个模型里输入向量都在一个特定的范围里。它用于softmax计算的分子和分母，每个元素都减去这个最大值。这样可以避免计算过程中出现溢出或精度损失。

局限性：
1. 使用范围有限：有些模型输入超出范围
2. 最大值选择不合适可能导致精度问题

## 4 Flat GEMM Optimization with Double Buffering
为扁平 GEMM 操作分配两个独立的共享内存缓冲区。一个用于执行 GEMM 操作，另一个用于加载下一个 GEMM 操作的新数据。这样计算和内存访问可以重叠进行
当 N 维较大时，采用双缓冲。较小的 N 维，可以使用 FastGEMV

关键问题：GEMV 或者 flat GEMM(bs >=1，比较小）操作在更小的 tiles 下（对齐到 Tensor Core 架构下的 8）如何计算。而不是更大的tiles：虽然掩盖了 memory latency，但是 GPU 利用率不高

Insight: 对于较大的 N 值，flat GEMM 受内存访问延迟的限制；而较小的 N 值，flat GEMM 受并行性限制

解决方法是两个缓冲区，这样一个在计算时，另一个可以做数据读取，这样掩盖仿存开销（前提是 shm 里足够放得下才行）。所以对较大的 N 值有用。较小的 N 值

## 5 Heuristic Dataflow with Hardware Resource Adaption

尽管，Tensor Core通常被视为GEMM操作的默认，GEMV更适合CUDA Core

1. 没有开源
2. 
