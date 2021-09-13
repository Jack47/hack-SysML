An error-compensated Adam preconditioned momentum SGD  algorithm.

1. 在解决的是什么问题？DL 模型训练里的通信开销很高的问题
2. 为何成功，标志/准是什么？跟非压缩的 Adam 对比：收敛性一样，减少了5倍的通信开销，达到 3.3倍的速度
3. 在前人基础上的关键创新是什么？提出 1 bit Adam， 做了实现，是 error  compensation 类型技术: 实现了自己的集合通信原语（不能用 nccl 的, 没有 all2all，而且无法压缩）
4. 关键结果有哪些？减少了通信开销，速度更快
5. 有哪些局限性？如何优化？是否能应用到 SGD 里？需要一段时间之后再启用 1 bit
6. 这个工作可能有什么深远的影响？

## 高效通信的分布式训练相关工作
1. 量化：Wangni
2. 去中心化
3. 异步通信
4. 梯度压缩。因为压缩过程可能有重要信息丢失，导致收敛速度变慢，所以后来又有了 error-compensated(误差补偿) 压缩策略

先使用 vanilla Adam 运行几个 epochs 作为热身，等variance 不变后，就开始压缩：此时 v 不需要更新，而momentum 上做 1bit 量化，只需要知道是增加还是降低

实现的压缩版本的 allreduce，有两个版本：

1. CUDA-aware 版本，使用了 GPUDirect 技术，需要 MVAPICH2-GDR 这类知道 CUDA 的库 => 需要有 IB 才能跑
2. 基本版本：可以和现有的 MPI 库一起使用，但是是在 GPU 和 CPU 间拷贝数据 => 任何有网卡的机器都能跑
## 收获
1. SGD 和动量(momentum) SGD 都是和梯度线性相关，而Adam 是非线性、基于梯度的
2. Adam 里的非线性部分：方差在经过 warm up 阶段后，变得稳定，可以在后续的压缩阶段里，当作固定的前提条件(fixed precondition) 来用。
3. 看看 1bit SGD: Seide 2014。近年来的研究表明 error-compensated 压缩，能达到无压缩的收敛速度

## 问题
1. 看懂里面的那些公式
2. SGD 和动量(momentum) SGD 的区别和联系。他们都是和梯度线性相关？
3. Vanilla/momentum SGD 是啥？
4. 为什么在 BERT 等算法里， SGD 不如 Adam 高效
5. 看看代码里怎么做到公式 5，其中的 error cancellation
