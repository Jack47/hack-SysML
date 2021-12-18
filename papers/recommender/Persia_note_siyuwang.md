<a name="nrjF4"></a>
# Quick View
<a name="Sz2nW"></a>
## Abstract
<a name="TRqO2"></a>
### Background
模型规模增长，巨大模型难以训练<br />Model scale trend：

- 1 billion(Google, 2016)
- 12 trillion(FB, latest)

Recommendation model charactorization：

- 99% embedding：memory intensive
- 1% NN：computation intensive

​

![image.png](https://cdn.nlark.com/yuque/0/2021/png/25595470/1639210613133-0e8845f4-d7e1-4476-90d0-a92dba2fc9f6.png#clientId=u7329049a-eb72-4&from=paste&height=412&id=udf439653&margin=%5Bobject%20Object%5D&name=image.png&originHeight=412&originWidth=473&originalType=binary&ratio=1&size=51473&status=done&style=none&taskId=u2e7ce324-6b30-428f-a3df-16957d0d9a1&width=473)
<a name="Myge7"></a>
### Motivation
提出一种有效的分布式训练系统，该系统从**优化算法**和分布式**系统架构**两方面做了精心设计

- Algorithm：提出一种Hybrid training算法，其中dense和embedding分别使用不同的同步机制
- System：设计并实现了系统Persia，能够支持上述的Hybrid training算法
<a name="JYXCf"></a>
## Experiment Quick View
主要是列benchmark结果，对比收敛加速比。为了highlight Persia在**系统**和**算法**上都有优势，作者分别展示了两组实验：<br />**实验配置**<br />**CTR BenchMark Tasks**<br />3个公开benchmark task和1个快手内部生产task。

- TaoBao-Ad
- Avazu-Ad
- Criteo-Ad
- Kwai-Video

cluster<br />8*8 NVIDIA V100 with 100Gbps network<br />models<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/25595470/1639212135344-5210b8e3-90e7-45e0-81b2-cce664bafaa5.png#clientId=u7329049a-eb72-4&from=paste&height=253&id=uf94a13f1&margin=%5Bobject%20Object%5D&name=image.png&originHeight=253&originWidth=451&originalType=binary&ratio=1&size=60920&status=done&style=none&taskId=ua2b8528f-3e3b-451c-8dc8-f1c51412492&width=451)<br />​

结果展示

1. 达到指定AUC时，经过的总时间对比

这项实验验证了Persia在系统上面的能力，Persia均先到达收敛线，并且速度可达7.12X<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/25595470/1639212156294-5de7c318-ca7f-4dec-8fd2-bf9c60081c20.png#clientId=u7329049a-eb72-4&from=paste&height=232&id=uc7202afb&margin=%5Bobject%20Object%5D&name=image.png&originHeight=232&originWidth=997&originalType=binary&ratio=1&size=110986&status=done&style=none&taskId=ue5ba1db9-36ab-4a1a-8ef7-f8163465d68&width=997)

2. 达到指定AUC是，经过总steps数对比

Persia新提出的Hybrid Algorithm收敛很稳定。<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/25595470/1639212169599-8a65947a-79f2-4764-a919-e46a91e13d2c.png#clientId=u7329049a-eb72-4&from=paste&height=173&id=u4fc17a5f&margin=%5Bobject%20Object%5D&name=image.png&originHeight=173&originWidth=952&originalType=binary&ratio=1&size=94486&status=done&style=none&taskId=ud4f14542-dafd-48cb-aa37-181d88d067a&width=952)

3. Scalability

随着GPU数量增加，Persia在四个任务上的扩展性很好。最后一个任务虽然异步比同步扩展性好，但是上面的实验已经其AUC偏低（上面的图中最后一个实验可以看出）<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/25595470/1639212728335-b8327489-3a0e-43f4-9aba-de8ed3e75648.png#clientId=u7329049a-eb72-4&from=paste&height=240&id=ub8071899&margin=%5Bobject%20Object%5D&name=image.png&originHeight=240&originWidth=963&originalType=binary&ratio=1&size=107861&status=done&style=none&taskId=u1cd7ab28-51ce-4810-88f0-0f561666eeb&width=963)
<a name="m4A1V"></a>
## Conclusion
贡献点

- 可以支持参数规模为100T大小的推荐模型
- 提出了Hybrid algorithm，能够既保证收敛，又能充分压榨计算资源，提升利用率

Benchmark

- 在公开的benchmark任务和快手的生产任务上，Persia的性能是目前SOTA的7.12X加速；
- 并且能够在Google cloud platform上做到100T的scale
<a name="Cnkfr"></a>
# Details
<a name="Ww5ij"></a>
## Introduction
<a name="rNJgS"></a>
### Background

1. 模型size，Racing to 100T，并且这是刚需
1. SOTA能支持最大的模型也就12T，Persia是第一个到100T规模的system：
1. 全同步和全异步都有问题：
   1. 全同步，同步开销太大，计算速度慢
   1. 全异步，计算爽了，收敛亏了
4. 于是Persia横空出世，设计出一种Hybrid Algorithm，并且系统开源了
<a name="tqdxb"></a>
### Contributions
Core：提出了Hybrid algorithm+异构分布式系统<br />Hybrid algorithm：

1. embedding：异步训练
1. dense部分：同步训练

异构分布式系统：<br />允许同步和异步同时存在优化模型，同时对GPU，CPU，存储等资源管理业做了些工作。<br />最后，已开源可复现。
<a name="Qi8H4"></a>
# Preliminaries
<a name="W5L7H"></a>
## Problem formulation
常规，略过
<a name="e56vt"></a>
## 剖析已有方法
XDL：Embedding上PS，其他在worker GPU上（本来不就该是这样？？）<br />Baidu PS：利用多级缓存，cache常访问的item
<a name="lhh15"></a>
# Hybrid algorithm
<a name="o94wP"></a>
## ASGD的两个观察
这是支持Hybrid algorithm的基本理论

1. Sparse access的部分，用asgd足够，不会有什么bias
1. Dense部分的，用asgd，其staleness会引入bias
<a name="XIwXf"></a>
## Persia的做法
![image.png](https://cdn.nlark.com/yuque/0/2021/png/25595470/1639216476858-992c7923-964b-45a1-848c-c3653079a0ee.png#clientId=u7329049a-eb72-4&from=paste&height=407&id=ua11b4201&margin=%5Bobject%20Object%5D&name=image.png&originHeight=407&originWidth=1007&originalType=binary&ratio=1&size=108552&status=done&style=none&taskId=u99bd51d7-d628-4431-810e-a30dc087bfc&width=1007)<br />上图中，理解时应注意，同一批sample的workflow顺序不可能打破，即GE->FC->BC->Dsync->PEG。因此Async和Hybrid中，GE和PEG能够并行是因为来自于不同的batch。

- Fully Async：不同batch间，连Dsync和Dense部分的计算都是异步的。上一批batch计算的grads更新时，此批batch的仍可能在计算，存在staleness；
- Naive Hybrid：同batch的Dsync和计算保证同步，但与不同batch的GE和PEG仍然异步；
- Persia：在上述Naive Hybrid之上，加了Dsync和BC的overlap。
<a name="l9hzd"></a>
## System Design
根据角色分了一些模块，如下图：<br />​![image.png](https://cdn.nlark.com/yuque/0/2021/png/25595470/1639217370628-b83528f1-67e5-4227-a83f-7e7c0d8d312f.png#clientId=u7329049a-eb72-4&from=paste&height=421&id=u6c24afb8&margin=%5Bobject%20Object%5D&name=image.png&originHeight=421&originWidth=488&originalType=binary&ratio=1&size=65269&status=done&style=none&taskId=u299588d7-e589-41f1-bcd4-7159e6e5357&width=488)
<a name="WWtIl"></a>
### 一些设计细节

1. Embedding Worker和NN worker上的Buffer cache

每个数据都有ID feature和非ID feature，一个要走Embedding worker，一个要走NN worker，最终都在NN worker汇合，汇合时需要能够再次将同一sample的数据拼凑在一起，靠的就是data loader上生成的ID以及两类worker上的buffer机制。<br />作用是为每个Sample找对应的worker：

- NN worker上的Buffer：非ID feature发送到NN worker上，最终要ID feature对应的Embedding worker的结果做对应。NN worker上就靠这个unique id做pull
- Embedding worker上的Buffer：记录在parameter从哪里来的，最后得把gradients push回去。
2. 显存管理和cache（不知道有啥用，可能是为了应对动态embedding而做的LRU）
2. 通信优化：
   1. NN部分的AllReduce通信，重在计算通信overlap，用的是Bagua；
   1. RPC通信优化：因传输的是连续空间的大Tensor，不想引入序列化开销。作为代替，使用了zero-copy serialization和deserialization的方法，直接对memory layout做序列化。
4. PS上的访问balance问题：先均匀shuffle，再平均放在多个ps上，能够缓解这个问题。
4. 通信压缩：
   1. 无损：其实就是多级索引。放弃对每个sample使用大的int64的ID，而对整个batch使用一个大的ID，每个小example直接使用index。
   1. 有损：FP16
6. 容错机制
<a name="AjjtR"></a>
## Hybrid algorithm收敛性证明
证明思路：只有embedding是stale的，最后推导出convergence rate不大于vainila SGD+staleness影响。而staleness只有embedding引入，其值为![](https://cdn.nlark.com/yuque/__latex/d34e305553178efca412cf7c0e2c737f.svg#card=math&code=%5Calpha%3D%5Cfrac%7B1%7D%7BN_%7Bemb%7D%7D&id=u5I79)远小于1，因此基本可以认为趋近于0，收敛性与vanila SGD相当。<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/25595470/1639220388931-b2d7ca3b-d75d-4a77-ae6c-c90f4a60eebb.png#clientId=u7329049a-eb72-4&from=paste&height=100&id=u57fe678f&margin=%5Bobject%20Object%5D&name=image.png&originHeight=100&originWidth=457&originalType=binary&ratio=1&size=7918&status=done&style=none&taskId=ubb4579c8-0590-48c4-abda-4aa367ef076&width=457)<br />​

​

​<br />
