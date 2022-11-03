1. 在解决的是什么问题？想让 self-attention 加速
2. 为何成功，标志/准是什么？尝试了独特的角度：让它使用 IO-感知的方法，利用好 GPU 层次结构的显存读写
3. 在前人基础上的关键创新是什么？ 减少 HBM 的读写次数，多用更快的 shared memory。基于 flash attention 还提出了更快的近似算法：block-sparse attention
4. 关键结果有哪些？速度快，因此能允许更长的序列在一些NLP的比赛里分数更高
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 介绍
虽然 transformers 越来越大，深，但是使用更长的上下文依然很难，因为自注意力模块核心的时间和显存复杂度都是序列长度的二次方。所以一个核心问题是让attention更快、显存更高效是否能帮助 transformer 模型克服他们在长序列(long sequences)情况下运行时和显存挑战。目前的 GPT3 被限制在了 2k 的长度。而 FLASH 可以长到8k，最长64k。这样还可以用到高分辨率的VIT上。

有很多近似的attention方法想减少计算和显存开销。包括 sparse-approximation， low-rank approximation，以及他们的组合。尽管他们减少到线性或者接近线性序列长度的计算开销，但是很多并没有明显的加速效果，因此使用不广泛。一个主要原因是他们集中在减少 FLOPS（可能跟加速不直接关联），因此忽视了仿存方面(IO)的开销

本文里，我们认为之前没考虑到的一个方法是：让注意力算法对 IO-感知，仔细审计对不同级别快慢不同显存的读写（比如快的 GPU 上的片上  SRAM和相对慢的 GPU带宽显存，或者 HBM，见下图1）。现代的 GPU 上，**计算速度比显存速度要快非常多**，transformer里的大部分操作都是仿存速度为瓶颈。IO-aware 算法对类似内存制约的操作上很关键，即当读取和写入
数据在执行时间上占大头的情况：例如数据库 joins，图片处理，数值线性代数(numerical linear algebra)。但是 Python接口的深度学习框架比如 PyTorch 或者 TF 没有暴露这种显存访问的精细接口。


尽管利用的技术（tiling & recomputation)都是已知的，但是依然有空间(2-4x)来加速 attentioin。需要使用 softmax decomposition。我们认为有两个原因导致虽然FLOPS增加（重计算），但是加速了：

1. softmax decomposition with scaling 虽然被很多 ML 算法人员熟知，但没被很多系统研究员知道，虽然 operation fusion/memory IOs 减少是系统/编译器社区的常备，但是对算法工程师而言却不熟悉。在 F 里，即需要 softmax decomposition，也需要 operation fusion 来达到加速和节省显存的效果
2. flash attention 里，算法是没变的，所以模型精度不受影响，甚至能提升（因为可以训练更长的序列）

附录 E.5 (NVfuser 1.12, AOT compiler from functorch, and TVM)里添加了自动做op fusion的baseline。Megatron 的比 这些快，而 flash 比他们还快。因为 softmax 被分解为了 local softmax。希望编译器领域的进展可以让未来加速/fusion成为可能。我们也在和编译器研究员一起自动化这些技术

与 Self-Attention Does Not Need O(n^2) Memory 之间的区别？(附录 B.5)。他们聚焦在减少总共的显存需求，利用的通用的 gradient checkpointing 技术，FlashAttention 简化了 backward （B.2, B.4)，获得了加速，节省了显存

与 FLAT 区别：它缺少 softmax decompossion 技术 (L151)，因此需要在整个行（或者多行）上计算 softmax。因此，flat 需要定制的有更大 sram 的硬件来放整个 key 序列，这个在 GPU 上不实用

要想让 automatic fusion 能把 softmax 也 fuse 了，需要先把 softmax 操作改为 decomposed 版本，再走 fusion 过程。希望未来编译器能支持这个特性。

重计算的开销：很快，因为输入都在 SRAM 里（为啥？），因为是作为 backward kernel 里的一部分。图2左侧展示了 pytorch 和 flash 的实现区别：虽然因为重计算多了13%的 FLOPs，但是因为减少了9.1倍的 IO，导致有 5.7倍的加速
F 对 batchsize 变化不敏感。比如 16，32，64上，都是 2-3x 的比 Megatron-LM 要快

MLPerf 里的最新实现是 vendor 里软硬件组花了6个月的实现。但我们的依然比他们的快

## TODO
1. Long range arena: A benchmark for efficient transformers
2. Efficient transformers: A survey.
3. Data movement is all you need: A case study on optimizing transformers
4. Memory hierarchy design. Computer Architecture: A Quantitative Approach(2003)
5. Roofline: an insightful visual performance model for multicore architectures(2009).
## 参考资料
1. [openreview](https://openreview.net/forum?id=H4DqfPSibmx)
2. [在 PyTorch 里的实现](https://github.com/pytorch/pytorch/pull/84771): 没有用 triton，用的cutblass
