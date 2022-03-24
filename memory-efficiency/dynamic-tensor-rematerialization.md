1. 在解决的是什么问题？训练时显存不够的问题
2. 为何成功，标志/准是什么？能训练3倍大的模型，速度影响很小
3. 在前人基础上的关键创新是什么？不需要提前规划，而是记录了访问的时间，动态计算出哪些需要换出去。发现效果跟离线效果差不多。核心是证明线上算法不需要提前知道模型的信息，就能实时产出一个特别好的 checkpointing scheme。这种在线方法能处理静态和动态图。
4. 关键结果有哪些？相比静态提前耗时小时级别进行规划的方案，本方案能搞定动态图
5. 有哪些局限性？如何优化？影响速度，能否提前重计算？micro-batchsize 下能用吗？
6. 这个工作可能有什么深远的影响？PyTorch 等框架应该自带这个功能

Rematerialization：在反向传播时重新计算中间值，而不是存储他们。其实就是 activation checkpoint 技术

根据图三，看到 DTR 在静态模型上的计划和 Checkmate的最优计划可以一比，甚至比过去的一些静态方法还快，静态方法可能需要花好几个小时才能找出方法，甚至不是最优的。最终发现 DTR 可以在静态模型上当作一种静态的 checkpointing 技术来创建 checkpointing的方案。这种 DTR 的用法不会额外产生运行时开销

为什么大家用 checkpoint(重计算）较多，用 swap 较少？因为相比较而言，一个算子 swap 的开销是重计算的两个数量级倍，可以参考 MegEngine DTR介绍里的那个表格

 图4: 
 
 DTR runtime overhead: 

1. Cost Compute: computing heuristic scores
2. Eviction Loop: comparing scores over tensors


## 启发
1. 可以提供一个指导加 checkpoint 的方法？产出静态 checkpoint 的布局图


## DTR 和 Capuchin 的对比
1. DTR 支持 dynamic tensor。Capuchin 里的 checkpoint 方法不支持，因为runtime 需要执行 profiling pass，然后假设计算图是静态的。这样才能固定访问 tensor的模式，然后制定出内存管理的策略
2. capuchin 里，recompute 是二等公民，swap因为用的 PCIe，并不跟 GPU 并发冲突，所以优先 swap 的
可能不能recompute 提前是因为实现复杂点，目前  pytorch 里默认用一个 stream，所以多个不同类型 kernel 是串行的，不会并发
而且显存用很多时，计算也达到峰值了，没有空间用来并发 recompute（记得论文里这么写的

## 参考资料：
1. [DTR OpenReview](https://openreview.net/forum?id=Vfs_2RnOD0H)
