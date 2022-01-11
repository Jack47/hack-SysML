1. 在解决的是什么问题？在 Data Parallel Training 场景下，用一个通信库，解决网络多样条件下，支持多种通信的模式(sync vs async, Full vs Low precision, Centralized vs Decentralized)
2. 为何成功，标志/准是什么？支持 allreduce 和 ps 两种模式下， Sync/Async, Centeralized/Decentralized, Full/Low precision 的组合下的通信模式
3. 在前人基础上的关键创新是什么？atomatic batching和通信的调度。除了标准的 SG 外还支持 Adam等。优化了 TCP 场景下的 all-reduce，
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？虽然提供了多种方式，但是没法帮助用户自动选择最合适的。只关注在数据并行
6. 这个工作可能有什么深远的影响？

## 启发
1. 看看如何 autotuning？

## TODO
1. 了解下 all-reduce 里的 ring all-reduce 阶段
## 问题
1. parameter server 里的多个 shards，是同一个 layer 的不同分片？如何能每个 shard 各自 拉取自己关注的 weights，然后forward、backward 呢？server 是按照什么规则合并梯度并更新 weights 的？
