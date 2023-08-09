https://vllm.ai/
核心还是 PagedAttention，比当时的 HF 的 Text Generation Inference 要快很多


* 利用 PagedAttention 来高效管理了 Attention 机制中的 key 和 value 这俩显存大头
* 持续 batch 了进来的请求

## 秘密武器：PagedAttention

LLM serving 中瓶颈是显存，在自回归的decoding 进程中，所有的输入 token 会产生 attention key 和 value，这些都在 GPU 里，用来产生下一个 token，这些 cached key 和value tensor 被叫做 KV cache。它的特点：

* 很大：LLaMA-13B 里就有 1.7GB
* 动态：大小取决于 sequence length，是高度可变而且不可预测的

所以如何高效管理这些显存就很重要。我们发现已有系统中 60%~80% 的显存都是因为碎片和过度预留浪费了

跟 OS 里 虚拟内存这里的 paging 机制类似。PA 可以在不连续的显存空间里，保存连续的keys和values。它把每个序列里的 KV cache 都切分到 blocks 里，每个 block 包含固定长度的 key 和 value 的固定数量的 tokens。在注意力计算时， PagedAttention kernel 会高效地识别并拿取这些 blocks 里的数据

因为 blocks 不需要内存连续，所以可以跟 OS 里的虚拟页一样灵活管理：可以把 blocks 当做 pages，tokens 当做 bytes，sequences 当做进程（进程之间可以共享相同的 pages)。一个序列里的持续的逻辑块，可以通过 block table 来映射到非连续的物理块上。

这种方法，肯定没有直接一大块读取效率高，但好处是显存就不需要那么大了。而且能让 LLM 这种需要上下文的场景下，复用之前的 K和 V，即重复的 K和 V 只保存一份
