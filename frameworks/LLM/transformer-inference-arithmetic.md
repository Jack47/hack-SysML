本文呈现的是几个 LLM 推理性能相关的原则，没有经验或者困难的数学。非常简单的推理延迟模型就可以是经验结果的很好的拟合。让我更好地预测并形成了更好的 transformer 推理的解释

## 目录：
### kv cache
解释了在推理的时候把 self-attention 向量缓存来带来性能收益，也讲了可能得收益和容量代价

在采样时，transformer 推理包含处理一个提供的 prompt/context（可以并行发生(这里指的 embedding? 比如 byte pair encoding），然后一个个 **采样** 额外的 tokens（这就是自回归的含义）。在采样的时候，transformer 执行 self-attention，这里采样时需要当前在序列(sequence)（无论是在 prompt/context 或者生成的 token 里）里的每个 item（指 token？）的 kv 值。这些向量被叫做 kv cache，即 past cache(GPT-2的开源代码里这么叫）。而 kv cache 的 shape 是 [batch, 2(k,v), num\_heads, seq\_len, features]

![](imgs/kv-cache.png)

上图里清晰看出来 context/prompt, sample(生成的），以及第一个采样 token 和 第二个采样 token，拿到这俩 token 的 kv 之后，就需要去走 transformer

问题：为什么叫采样 token？

目的是避免每次采样token时，都重计算这些向量。有了已经计算好的 k、v 的值，我们可以以存储开销来节省大量的计算开销。对于每个 token，需要存储的字节数量：
```
2*2*nlayers*nheads*dhead
```
相当于每个 k or v 的维度是 [nheads, dhead] # 那 dhead = seq_len*features?

> 如何计算一个矩阵乘的 flops？
> 1. 对于矩阵(m,n)和向量(n)，是 2mn
> 2. 对于矩阵(m,n)和矩阵(n, p)，是 2mnp
> mn 意味着很多，而2倍来自于一个矩阵乘法是由两个算子组成：乘法和加法的操作


在 token embeddings 里乘起来的权重是 Wk, Wv 属于 R(dmodelxdmodel)，因此每个 token embeddings 是 te 属于 R(1xdmodel)。所以计算所有层的 k 和 v 的 flops 是：

(dmodel 就是上文里的 features？）
```
2*(2*nlayers*dmodel^2)

```
这意味着对于 52B 参数量的模型（比如以 Anthropic 为例，dmodel=8192，nlayers = 64），那么 flops是：

```
2*2*64*8192^2 = 17,179,869,184
```


### capacity
 kv cache 的存储代价，模型权重的存储代价，对于性能而言，capacity 意味着什么
 
### model parallelism
解释 TP 来清楚地描绘通信的代价

### latency 计算
给推理速度计算出了天花板

### batch sizes
讨论了 batch size 对性能的影响，以及什么 bs 是最优的

### flops counting
浏览了 transformer blocks，识别出哪些操作对 flops 速度有影响

### intermediate memory cost
激活值如何占用内存，内存带宽看起来是什么样

### 和真实的 benchmark 做比较
我们计算的和 NV FasterTransformer 里的 benchmarks 报告作比较，得到差异
