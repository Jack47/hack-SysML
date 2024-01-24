System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models

可以看到能 scales 到 1M 的 token （随着GPU数量从4到64)，而且在 **4倍** 长上下文情况下，还能训的快2.5倍

几个挑战：
1. DP，TP和 PP 都不能帮助解决 sequence 纬度的 scaling
2. 已有的 sequence 并行方法不高效，因为有显存和通信的低效
3. 已有的方法使用方面有限制：侵入时，重构易出错

## 摘要
书级别的摘要，多模态下的高纬度的输入，长推理上下文。

针对上述三个挑战，那就是想办法把 sequence 纬度在多个 GPU 上进行拆分，各自计算完再聚合到一起

**关键特性**

1. 4x 更大的序列长度，可以训超过百万的 tokens（是总token还是单次最长的情况
2. 相比而言通信占比降低了10倍（怎么做到的？），导致吞吐提高2.5倍，175TFlops/GPU，超过 54%
3. 完全通用的，而且实现无感的：支持 dense 和 sparse attention，可以和 fa v2 一起工作
4. 支持大模型训练：可以和 ZeRO-3 共用，不仅支持长序列，还能支持大模型
5. 很好使用，移植，对已有系统需要更少的代码改动
