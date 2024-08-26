![](imgs/rlhf-iteration-breakdown.png)

上图里很好地说明了 RLHF 由三阶段组成：actor generation, Critic\Ref\Reward 来评估，最后 actor 和 critic 来训练

同样的一个模型，模型分布在每个 GPU 节点上，使用同样的并行策略，比如 DeepSpeed-Chat 那样。这样并行的太过了，会导致潜在的同步和通信的 overhead （亮紫色的块）。这不就是我们的场景？原因是更大的并行度（无论tp、pp或者 dp）下，通信开销更大。

一种可能的方式是不同的模型，分配到不同的 GPU 上，模型并发执行，比如 OpenRLHF。然而，我们观察到非对称并行通常导致 GPU 资源利用率低（比如灰色区域），因为任务之间是串行依赖的

ReaL 的核心是动态在不同 GPU 间分配模型参数，来提高整体训练效率。
针对 Generation 来使用 Pipeline，训练阶段使用 TP，让这些操作可以在更低的并行维度行进行（节省通信开销），目的是最大化 GPU 利用率，减少冗余存储。

![](imgs/RealHF-kernel-execution-time.png)

这个图里来看，actor 模型越大，通信开销越大？这个通信开销主要是来自 TP 和 PP？这个图也有点乱，左边是 REALHF，右边是 REALHF-Heuristic。但是图的横轴又不一样。

图10 来看，RealHF 的主要时间下降来自于通信开销的降低。
1. 特定阶段使用特定的并行策略
2. 不同阶段并行走

## 8.4 Case Study

图13，可以看到当 Actor Train 结束之后，立马就可以并发地开始 critic train和下一轮的 actor generation 了。


TODO：
1. 

别人提交的 issue：https://github.com/openpsi-project/ReaLHF/issues/10


