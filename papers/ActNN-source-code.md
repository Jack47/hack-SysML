## 问题
1. 用贪心求解的是要用的 bit 数？它的限定是拿到 sample 了，就能算出来？
2. cuda kernel 里写了 if 条件？是否会影响效率
3. 一些函数的用法，如： atomicOr
4. 哪里是运用不同L的地方
quantization cuda kernel 实现在[quantization_cuda_kernel.cu](https://github.com/ucbrise/actnn/blob/9026b5fe8c3115a326c03a726a92ab87cf176d61/actnn/actnn/cpp_extension/quantization_cuda_kernel.cu?plain=1#L25)

## 看源码的疑问
1. 里面的混合精度 mixed_precision 是什么意思？针对每次 sample 和 每层，**适应性**选取数值精度

为何在 convnd 的实现里，在 run_backward 里需要实现 cudnn_convolution_backward 呢，里面还涉及 dilation 矫正。我的理解是不涉及到计算的。并不会把有损量化做纠正。发现这个只是重新链到了 aten 里的实现，不知道为啥要这么搞。这里的参数都是原生 PyTorch 里支持的。

unpack\_single\_precision\_kernel
它里面怎么知道边界？比如有多少数字要解压缩出来。看解压本身过程很简单，就是缩放

看看 mixed precision 怎么做的？其中哪里体现了动态调整 bits 的地方

每个 layer 里用的方法有何不同？

batch normalization 里做了些什么？



## 流程
1. 每次在 run\_forward 之后，把激活值量化一下，保存到 ctx 里
2. 在 run\_backward 需要用到激活值时，反量化出来，然后计算实际算子的 backward 过程

## 目标：看懂 quantize_activation

## 目标：看懂 quantize_activation

## 基本概念
Warp shuffle: \_shfl\_down_sync

pytorch tensor  slice

block, thread
