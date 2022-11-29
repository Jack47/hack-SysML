1. 在解决的是什么问题？希望在便宜的spot实例上，进行高效、抢占安全的大规模训练
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？


它比普通的训练的 pipeline 要长 1.5倍，因为：

1. 要有 redundant layers
2. 吸纳潜在的pipeline调整

但是因为spot实例很便宜，所以总体看下来还是划算的

本质是除了要弹性（机器可以被拿走），还要兼顾从抢占里恢复的效率，因为可能会被频繁抢占。它实现原理就是用一部分冗余计算来提供 resilience 和快速恢复的能力.

https://github.com/uclasystem/bamboo

bamboo 为了避免一下子被抢占的机器太多，所以会把相邻的机器放到不同的区域里，减少相领的机器被抢占的代价

原理跟p2p的原理类似，就是用冗余来保证鲁棒性。一个节点挂掉，其他节点那里有数据，可以顶上继续计算

Strawman #1: Checkpointing

Strawman #2: Sample Dropping
也叫做 elasticbatching ，因为丢掉 sample等价于让训练的iter里batchsize是弹性的

### 5.2 Schedule Redundant Computation

Eager FRC:

Lazy BRC and Recovery
这个并不是必须得，只有当发生机器被拿走时，才需要拿到上面FRC的激活值，然后把挂掉机器的BWD计算出来
## 启发
1. 在 7 Related Work 里，提到了好几个方向： Elastic Training, Exploiting Spot Instances, GPU Scheduling 三块，有对应的参考文献，可以看看
2. 在 DDP 下，如何做？因为DDP 里没有 bubble time 来调度 RC。通过多启动一些机器，比如1.5倍，来让每个节点处理更小的batch。
3. checkpoint 也可以异步去做，做频繁一些。
