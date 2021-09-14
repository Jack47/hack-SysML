## 总结
1. 在解决的是什么问题？如何训练大模型，解决内存不够的问题
2. 为何成功，标志/准是什么？ 能支持更大的模型，效果很好，比如1k个节点上支持 1T 的参数训练
3. 在前人基础上的关键创新是什么？去掉了DP 模式下的内存冗余，而且把内存里生命周期短的 activation 和gradient等可以进行挪动来避免内存碎片
4. 关键结果有哪些？ 能训练更大的模型
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 一些数字
Adam 里保存两类数据：
1. time averaged momentum
2. variance of the gradients to tcompute the updates.

尤其是在混合精度训练下，优化器所耗的内存会更大：AMP 下，forward 和 backward 时的激活值和参数都是 fp16 的，这样能用起来 tensor core 加速。但是优化器里是混合的：fp32 的参数和其他所有优化器状态

t参数的模型，DDP内存用量：

fp16 的**参数**和**梯度**：
parameters：2t 
gradients: 2t

优化器里需要存储： Adam optimizer states：12 t . 其中包括了 fp32 的参数、其他优化器状态:动量(momentum)和方差(variance)

## 7 ZeRO-DP 里的通信分析

zeRO-DP 里使用 OS+P，没有额外的开销
一般 all-reduce 分为两步：
1. reduce scatter: 分散到每个进程里做自己负责的那部分的 reduce 工作
2. all-gather: 每个进程 gather 所有其他进程 reduce 好的数据

而 zeRO-DP 里也是这样，reduce scatter 。这里没太看懂，为啥没有额外的开销，跟原来方法的不同在那里？为啥说上 OS+P，能节省8倍开销


## 问题：
1. 节省了内存后，训练的速度也会变快？ 可能原因是数据分片，计算梯度、参数，计算优化器状态更新也都分片了？(并不是, 只是存储方面分片了，没有冗余的数据）而且内存管理更简单，所以管理内存的计算量也减小？
2. ZeRO DP 是不是只能用在 NLP 里呀，它的目标是优化NLP 里内存大头：Adam 优化器里的状态。否，也可以用在 SGD 里。因为 SGD 里 AMP 下，需要保存：
3. P2 里为什么说 ZeRO-DP 里的 Pos 节省 4倍内存，Pos+g 节省8倍？不是跟机器数量相关的么？
4. 为啥 Parameter Partitioning 的内存节省跟机器数量(DP degree) 相关呢？
5. ScatterReduce 是啥操作？ 是两个步骤
6. ZeRO 是不是除了节省内存外，还能提高训练效率？并不能
## 

