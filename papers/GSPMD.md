## 问题：

1. 是否给予启发，比如不同模型用哪种方式好？
2. 



自动做并行，包括：数据，同一层内的并行，空间分片（spatial partitioning)，权重分片更新或者优化器状态分片和流水并行

它是基于之前 GShard 和 一个新的 Single Program Multiple Data属性，作为 ML 负载的通用并行方式。 之后，