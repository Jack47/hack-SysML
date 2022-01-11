1. 在解决的是什么问题？在 Data Parallel Training 场景下，用一个通信库，解决网络多样条件下，支持多种通信的模式(sync vs async, Full vs Low precision, Centralized vs Decentralized)
2. 为何成功，标志/准是什么？支持 allreduce 和 ps 两种模式下， Sync/Async, Centeralized/Decentralized, Full/Low precision 的组合下的通信模式
3. 在前人基础上的关键创新是什么？bucketing时，会把模型里的参数、梯度都放到一块，这样就能以块为单位进行 allreduce 和 update（但 ddp 好像也有？），atomatic batching和通信的调度。除了标准的 SG 外还支持 Adam等。优化了 TCP 场景下的 all-reduce，
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？虽然提供了多种方式，但是没法帮助用户自动选择最合适的。只关注在数据并行
6. 这个工作可能有什么深远的影响？

Bagua 的目标是：提供灵活的抽象来支持这些多样的训练算法，提供自动性能优化的框架，无须嘉定特定的通信模式，比如 DP-SG。

为了减少通信的量级，有量化、稀疏、sketching和 error compensation.

为了摆脱 latency bottleneck，提出了非集中式的通信方法

localSGD 是为了优化通信的次数

为了去除同步的 barrier，这个在非常大规模的 worker里有straggler 情况下，会是问题，提出了异步的更新方式。

## 3.
bagua的执行过程分两阶段：

1. Profiling 阶段: 先跑一个来回，bagua 会记录所有通信函数的开销，是没优化的版本。然后自动做两个优化：1. (Bucketing) groups layers into different buckets, whose communication will happen all at once(不跟ddp 一样么，目的是对all-reduce这类大通信量高效的通信库友好). 2. (Flattening) 重新安排模型中层和梯度，让同一组的放在相邻的空间，来取得**更好的局部性** 3. (scheduling) 调度什么时候做每个 bucket 的通信，让其与计算重叠
2. 执行阶段：执行优化

### 3.2 communication primitives
Decentralized: 梯度不在所有worker间进行同步，而是让每个人和左右邻居(可以是固定拓扑里的两个邻居 ，也可以是随机函数来多个)同步


### 4.3 Bagua 中的 Trade-off
收敛性：1-bit Adam 在 VGG16 上无法收敛，在几个epoch之后炸了。没有一个方法是在不同workload 上的万金油

### 4.4 网络条件
带宽较低时：通信压缩算法更好

延迟较高时：decentralized 算法更好

### 4.4 系统优化的研究

主要三类优化：

1. 通信和计算重叠
2. Tensor的融合和扁平化。让有**更多小 tensor **的情况更高效
3. 层次化通信: 多主机间的通信加速

## 启发
1. 看看如何 autotuning？
2. 如何找出集群中的 stragglers ？
3. Decentralized sync，是说延迟比较高，所以先用自己的梯度来更新参数？之后会再次同步吗？
4. 试试 Hierarchical Communications? 区分机器内和机器间，优化通信实现(可能会改变通信的语义)
## TODO
1. 了解下 all-reduce 里的 ring all-reduce 阶段

## 问题
1. parameter server 里的多个 shards，是同一个 layer 的不同分片？如何能每个 shard 各自 拉取自己关注的 weights，然后forward、backward 呢？server 是按照什么规则合并梯度并更新 weights 的？
2. bagua 的核心是训练时用的梯度更新算法，所以必须把通信和优化器结合到一起优化？
