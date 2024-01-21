
## 1. 为什么数据会影响 context scaling

上下文能力在非常大的应用比如多个 pdf 里的问答，一辈子时长的对话历史编码和 repo 级别的代码理解等领域


## 2. Summary of engineering basics

想达到10万的上下文，直接暴力就可以。核心技术是 FlashAttention

上采样长上下文有多种方式。最基本的是增加比 pretrained 长度（比如 LLAMA 2 会从 4K 增加到 80 K)更大的数据的比例，但这个会改变数据的混合（比如一些特定领域比如书籍就是更长的）。更多的方法可以考虑同时控制 domain 和 长度
上述最简单的方法下，上采样超过 4k 的数据，然后继续训练大约 10B tokens 就可以达到相对比较好的 context scaling 的性能，重现 LLaMA2 和 Claude 2.1 里的 power-law 的scaling 曲线。

对长依赖建模更像是模型在 4k 长度上训练后已经具备的能力。100k 更像是适配而非能力注入：模型在 200M tokens 时就可以把 loss 收敛到相对低的状态

## 3. Summary of capability balancing
不管长下文数据怎么上采样，C4，CC 和 StackExchange 下的 dev loss 不怎么显著变化

提高并不会在不同 domain 之间迁移，或许在短和长的文本之间存在矛盾



## 参考资料：
Extending Context Window of Large Language Models via Positional Interpolation(2023) (LLaMA Long Paper)

YaRN: Efficient Context Window Extension of Large Language Models


