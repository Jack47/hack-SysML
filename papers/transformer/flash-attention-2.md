https://princeton-nlp.github.io/flash-atttention-2/

https://tridao.me/publications/flash2/flash2.pdf

GPT-4 context length 是 32K，而 MosaicML's MPT 是 65K，而 Anthropic 的 Claude 是 100k。怎么做到的？而书生只有 8k。

FlashAttention 第一代虽然已经比当时发布的 baseline 要快2到4倍，但跟优化过后的 GEMM 操作比，还是不够快，只有理论最大算力的 25-40% FLOPs/s（比如 A100 上最大是 124TFLOPS）。它比之前版本快2倍，达到了 A100 上的 225TFLops( 72%）。

## FlashAttention-2: 更好的算法，并行度，工作的切分

### 更少的非矩阵运算
A100 上 FP16、BF16 matmul 能达到最大的理论吞吐是 312 TFLOPs/s，而非 matmul 的 FP32 只有 19.5 TFLOPs。也说明 非矩阵 的 FLOP 是比矩阵的 FLOP 要慢 16倍的。重写了 softmax trick，来减少 rescaling 操作的次数。

### 更好的并行度
第一代里使用的是在 batch size 和 head 数量两个维度上进行并行，即使用一个 thread block 来处理一个 attention head，因此有 batch_size * number of heads 数量个 thread blocks。每个 thread block 会被调度到 一个 SM 上，而 A100 上有 108 个 SMs。因此当数量很大（ >= 80) 的时候，效率比较高，因为用了几乎 GPU 上大部分的资源。

在长序列的情况下（通常意味着更小的 batch size 或者更少的头），为了高效实用 GPU  上的 SMs，需要额外在 sequence length 维度上进行并行。这个会显著加速。

### 更好的工作划分

## 新特性：head 支持到 256，多 query attention



