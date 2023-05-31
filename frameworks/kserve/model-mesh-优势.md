## Modelmesh 的优势：

1. 模型的信息有变化，能及时感知到
2. 全局的显存大小知道，所以可以预估出它能容纳的模型大小。这样能更充分加载模型到显存里，减少切换的时间。比如 `9台主机*8卡*每张卡容纳3个开源模型=216个模型`。推理时，只有发生了 cache miss 才load。这样控制面能分离出来

3. 扩容决策是每6s执行一次



## 我们需要知道哪些指标

扩容：看 rpm 或者 latency

缩容：最近没有被使用

问题：这样会不会很快扩上去，但是缩不下来？

### Instance-models

**来自 runtimeCache：**

count, capacity, used-capacity,



**来自 instanceInfo():**



lruTime,  configured-loading-threads, total-loading-threads

Request-per-minute, instance-version, location?, zone, labels



#### 生命周期：

janitorTask：（有个 rateTrackingTask）

​	cacheChanged、loadLocal（是cache miss之后的逻辑）、invokeModel、CacheEntry、handleInstanceTableChange（instanceTable变更触发）、establishBackgroundTasks()

​	扩容策略：按照 当前rpm来等比例扩容，那扩容时选择哪些节点？

怎么知道当前GPU上空闲了多少容量？模型的运行时开销怎么预估的？



Prometheus 里也放了一份(所以跟zk里一样的，但不是最新的，而是平均的）： metrics.logInstanceStats()