虽然 bs 维度可以扩展，attention 很快就会在相对小的长下文里遇到瓶颈。因为读取的显存可以随着 bs 增大 scale，因为只跟模型大小有关。

flash-decoding 可以加速 attention 的推理过程，对于非常长的上下文可以加速 8 倍。核心思路是尽可能把 **key 和 value 都并行加载**，然后单独 rescale 并最终结合前面的结果来维护处正确的 attention 输出。看这里的[图片会非常直观](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)

## Multi-head attention for decoding
在 decoding 阶段，每个新产生的 token 都需要和前面所有的 token 结合，来计算： softmax(q@k.transpose)@v

这个操作在训练阶段已经被 FlashAttention （v1和v2）优化过了，瓶颈是显存读取和写入中间结果的带宽(比如 Q@K^T)。然而并不能直接用于推理阶段的优化，因为瓶颈是不同的。训练阶段，FlashAttention sh 是在 bs 和 query length 维度并行的。而推理阶段，query length 通常是1：意味着如果 bs 比 SMs 的个数还小(A100 是 108），这个操作只会用到 GPU 里的一部分。而长上下文的情况，正好就是这样，为了能在显存里放得下，bs 必须得小。如果 bs 只有1，那么 flashattention 只会用到 1% 的 GPU

FlashAttention 在query 块和 bs 上并行，所以在 decoding 阶段（bs小，而query通常是1）
## 

适用范围有限：主要针对 LLM 的注意力机制进行优化，对于其他类型的模型或任务，优势不明显；实现复杂度较高，需要对键值对切割，并行计算注意力，缩放和组合注意力结果等步骤
