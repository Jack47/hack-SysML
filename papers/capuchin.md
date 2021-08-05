## 思考
1. 在解决的是什么问题？高效的内存使用，在现有 GPU 硬件上运行更大的模型，更大的batchsize
2. 为何成功，标志/准是什么？ 支持了更大的 batchsize 
3. 在前人基础上的关键创新是什么？ 内存管理不是常见的 layer 维度，而是更细粒度到 tensor 级别，是计算图无关的。而且是在运行时根据profile的结果来决定应该用 swap 还是 recomputation，其中 evict/prefetch，recomputing 的时机都是都是动态决策出来的。
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 1. 介绍
Capuchin的设计基于两个关键观察:

1. 所有的深度学习框架都是基于数据流图的执行模型。所有步骤都是基于 tensor 上的操作。所有传统程序里的重用和局部性原理都适用于 DL 
2. DL应用的属性确保了我们方法的有效性：训练过程由成千上万的迭代组成，各自时间有很清晰的界限，在每个 itertaion 间，tensor 的访问是有规律和重复访问的模式

## 2. 背景
内存使用：

训练时内存主要有三部分：输入（激活值），权重，和其他比如计算卷积时需要的值，还有优化器里的状态，比如 SGD 里的二阶动量

### 2.2 DL框架执行模型的差别
Eager Mode：不构建图，直接从头开始执行。这样部署和debug 新模型比较方便，在学术界用来设计新模型时用的很多。TensorFLow 2.0 里默认就是这个。

Graph Model: 先构建一个计算图，图里的执行策略时 lazy，需要时才执行

## 3. 观察和动机
### 3.1 静态分析(vDNN, checkpoint)的局限
1. swap in 的时候，可能时机选择不合适，导致跟其他算子重叠的比例不够高。 swap 的时间取决于换入的大小及 PCIe 带宽，而某层的执行时间是由 GPU 型号和当前layer 所需算力和输入规模。
2. 同一层上同样类型的执行时间会变化。比如不同 GPU 上，不同 batchsize 下。举例，并不是说所有 CONV 都不能 recomppute。有一些计算大地家非常小，3ms左右，是可以重计算的

### 3.2 机会：Tensor 访问的常规模式(regular pattern)
不同 tensor 的多(两)次访问(forward, backward)之间间隔不同。而且不同tensor 之间的访问时间间隔也不同

## 4. Capuchin 内存管理机制
### 4.1 两个设计目标：

1. 最大代价去最小化额外开销。因为会有成千上万个 iteration
2. 足够通用，在不同 DL 框架里不需要太多代码改动

怎么没有提升多少性能之类的目标?

### 4.2 Design Overview
第一个 iteration 里进行执行的测量(measured execution)，第二个开始的 iteration 里使用自己的策略，算是被指导下的执行(guided execution)

提供 passive mode:按需进行 swap，比如 out of memory 了，就执行 swap。跟硬盘到内存，虚拟内存机制非常像。

在 measured execution 里，Capuchin 保存了 tensor 访问次序，以及 access count，timestamp，operation that produced a tensor and  its input tensor 来为重计算作准备。为了提高效率，在访问完一个 tensor，Capuchin 可以主动提前主动发起换入/换出这个内存。这样相比于有需要才切换，能够掩盖切换的开销。
 
 关键问题是如何选择合适的时间点来进行操作？需要集合 swap 和 recompute 的代价。
 
 定义几个概念：
 
 1. evicted-access: 是访问 tesnor，然后出发换出操作的访问
 2. back-access：是换出之后的第一次访问。此时 tensor 可能在，也可能不在 GPU 显存里。如果没有主动的(proactive) tensor 重新产生，tensor 肯定不在 GPU 里，这样就增加了重产生的开销。我们可以在别的 tensor 访问时出发重产生当前 tensor 的操作。叫做 in-trigger 。可以是一个 tensor 的 evicted-access 或者 back-access 来触发。如果没有指定，就认为访问失败（不在 GPU ） 后触发 re-generate
 
 上述两者之间间隔越长，越可以掩盖重新产生(swap or recompute) 的代价。1
 
 后面主要回答一下几个问题：
 
 1. 如何预估 swap 和重算的代价
 2. evict 哪些 tensor
 3. 什么时候 evict 和 重产生
 4. 如何重产生（swap 还是 recompute）
 
一般情况下，swap 时，我们应该增加计算和 swap 叠加的时间

recompute 时，选择代价更小的

所以关键是预估出 swap 和重计算的代价。
## 启发
1. 能够把 overlap 思想结合到 actnn 里？让解压缩提前做，这样解压缩和backward 同时进行





