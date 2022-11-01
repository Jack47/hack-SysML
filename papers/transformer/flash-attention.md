1. 在解决的是什么问题？想让 self-attention 加速
2. 为何成功，标志/准是什么？尝试了独特的角度：让它使用 IO-感知的方法，利用好 GPU 层次结构的显存读写
3. 在前人基础上的关键创新是什么？ 减少 HBM 的读写次数，多用更快的 shared memory。基于 flash attention 还提出了更快的近似算法：block-sparse attention
4. 关键结果有哪些？速度快，因此能允许更长的序列在一些NLP的比赛里分数更高
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 介绍
虽然 transformers 越来越大，深，但是使用更长的上下文依然很难，因为自注意力模块核心的时间和显存复杂度都是序列长度的二次方。所以一个核心问题是让attention更快、显存更高效是否能帮助 transformer 模型克服他们在长序列(long sequences)情况下运行时和显存挑战

有很多近似的attention方法想减少计算和显存开销。包括 sparse-approximation， low-rank approximation，以及他们的组合。尽管他们减少到线性或者接近线性序列长度的计算开销，但是很多并没有明显的加速效果，因此使用不广泛。一个主要原因是他们集中在减少 FLOPS（可能跟加速不直接关联），因此忽视了仿存方面(IO)的开销

本文里，我们认为之前没考虑到的一个方法是：让注意力算法对 IO-感知，仔细审计对不同级别快慢不同显存的读写（比如快的 GPU 上的片上  SRAM和相对慢的 GPU带宽显存，或者 HBM，见下图1）。现代的 GPU 上，计算速度比显存速度要快非常多，transformer里的大部分操作都是仿存速度为瓶颈。IO-aware 算法对类似内存制约的操作上很关键，即当读取和写入
数据在执行时间上占大头的情况：例如数据库 joins，图片处理，数值线性代数(numerical linear algebra)。但是 Python接口的深度学习框架比如 PyTorch 或者 TF 没有暴露这种显存访问的精细接口。



## TODO
1. Long range arena: A benchmark for efficient transformers
2. Efficient transformers: A survey.

