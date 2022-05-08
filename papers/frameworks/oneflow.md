1. 在解决的是什么问题？现有 DL 框架在大模型上使用和实现复杂的问题
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？ SBP 的分布式抽象和 actor 模型来简化依赖和执行的运行时关系: 能管理资源限制、数据移动和计算三种依赖
4. 关键结果有哪些？简单、整洁，效率比其他要高
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 介绍
最好情况下，一个 DL 分布式框架应该：

1. 给定任意的并行方案后，能自动生成物理的执行计划，减少手工编程的代价
2. 更高级的需求是：框架能找到在任意NN网络结构和硬件组合下，最佳的并行策略

但，已有框架甚至都不满足条件一，即灵活支持多种并行策略。这是我们想在本论文里解决的问题：创新地重新设计DL框架

一些新兴的开源项目开发了专用的系统或者定制了库来更好地支撑模型和pipeline并行。例如，点击预估里的 HugeCTR，Megatron-LM和DeepSpeed 面向 大规模预训练 NLP 模型。InsightFace
是给大规模人脸识别的模型并行。但这些都是给特定应用定制的，并不能组合起来来形成一个通用的方案

也提出了一些插件来增强一些主流的DL框架来至此更复杂的并行策略。

Tensorflow： Mesh-TensorFlow, GShard 提供 API 来做并行。GPipe

PyTorch: PipeDream

由于现存框架设计之初没有预见到如此复杂的并行，导致在上面增量开发会带来显著的额外系统开销，需要研究员花费大量时间

## 2 背景和动机
一个 DNN 通常被表达为一个**逻辑**的op组成的计算图，通常是手工或者自动被编译器转化到由优化后的kernels组成的**物理图**来在运行时执行。分布式训练中包含数据（参数、梯度、激活值）
的通信。由于主机间带宽是设备内部的一到两个数量级的差异，所以 DL 框架应该把数据搬用当作跟计算一样的一等公民

### 2.1 把负载划分到空间领域(Spatial)

比如上图中的三层的网络，虽然从编译器输入角度来看只有三个算子的逻辑计算图，它在编译后，实际运行时的硬件上跑，可能是上图中下半部分那样复杂：f1和f2算子是2路数据并行，执行在 d1 和 d2 上，而f3算子因为太大而无法在单机上执行，
被切分为两路的tensor 并行：在d3和和d4上模型并行。图里蓝色部分里，有一些是通信的算子：g(allgather), s(reduce-scatter), r1和r2都是all-reduce 操作

![](./imgs/logical-physical-graph.png)

这个图不是很容易看懂：

1. 蓝色的是既有输入，又有通信算子
2. f的下标，无法区分出是数据还是模型并行
3. 通信算子，没有交代为啥是 all-gather、reduce-scatter
4. c31、c41 为啥放上去？r2是reduce完，再去更新参数？

### 2.2 把负载划分到时间领域(Temporal)

指如何安排算子来让硬件利用率最大化。通常性能提升的最好方法之一是让通信和计算尽可能重叠。

同一批次的 forward 和 allreduce 不能同时进行，但是数据加载和前处理可以和上一批次数据同时进行。反向传播和all-reduce 操作也是可以重叠的

### 2.3 管理复杂依赖
主流的 DL 框架里，数据和控制依赖都是用执行图里的边来表示的。在每个op执行结束后，调度器会更新剩余算子的依赖，找出依赖都被满足的算子来执行。

**由于资源共享而引起的依赖** 

![](./imgs/deadlock-example.png)

调度器需要决定合适的执行顺序来避免多个算子共享同一个资源情况下导致的 OOM 错误或者死锁(主要是通信和计算的共享，因为目前 pytorch 里计算流是默认流)。
考虑图2里的例子。M1和M2是两个数据搬运的算子，服务于同一个设备上的两个计算操作：O1和O2。O1和O2不依赖于对方，而O1相比O2需要更多设备显存来执行。M1
和M2自身也需要一些设备内存来存储搬运的数据。M1和M2执行完之后，剩余显存只够执行O2，但此时O1和O2都在调度器的ready集合里。如果O1先执行，显存是不够的。
而系统可能会报 OOM，或者阻塞住调度线程(不太理解，应该是阻塞住当前这个O1操作把？)，而后者会导致死锁。为了避免这种风险，框架最好提前指定执行顺序（比如
在 TF 里添加**控制**这种依赖）。如果系统使用流水来重叠数据移动和计算，这个问题会变更严重，因为M1和O1可以同时执行：M1搬运下一片内存时，O1在计算上一片数据。
在编译期**规划好资源**，在运行时做好**流控**是执行稳定所必须的。对于动态shape，应该是不好做提前规划的

**由于数据搬运而导致的依赖**

![](./imgs/interaction-callback-scheduler.png)

已有的DL框架没有把数据移动当作图里的正常操作。因此，数据搬用和计算之间的依赖并没有被表示在计算图里。例如，TF 把节点内的数据搬运包装在了 callback 函数里，
需要时插入。因此，计算/控制依赖在图的边里表达，而数据依赖在回调函数里。在图3里，O2被包装在O1完成后的回调函数里。然而当O2有其他依赖，比如需要其他op的输出或者
控制依赖(control dependencies)，此时O1完成并不能就执行O2。此时回调函数应该告诉调度器O1执行完：然后调度器返回其他依赖也完成了，O2可以立马执行；否则O2
需要被插入到等待队列里。将来所有依赖都满足后，再调度。（那不能就直接放到等待队列里，让调度器判断？） 得看看这种情况出现的概率大不大？
上述例子里，框架需要把内部调度器暴露给用户来让插入的回调函数能正确和调度器交互（图里判断是否ready的过程）
但目前 DL 框架不太支持这种操作，没有哪个 DL 框架给用户暴露了调度的接口。理想情况下框架应该把所有依赖显示地在图里OP间呈现(包含数据移动)。一旦达成，在运行时执行图
就会被极大的简化。

### 2.4 总结
设计了 OneFlow，一个编译器可以自动产生一个物理的图来执行数据、模型和流水并行。编译器支持完整的分析各种依赖关系（比如资源、数据搬运和计算）。如何做到支持共享资源的依赖判断呢？如何支持数据里动态shape？如何知道计算时的资源有多大?
而且我们基于actor model 设计了简练的执行时，能够用一个一致的图里actor之间消息传递来实例化所有的依赖关系

## 3 编译器

假设每个算子都被赋予了一个属性叫 placement，标示在哪个节点（物理机器）和设备上，这个逻辑的算子会被部署上去。类似地，一个全局 tensor（一个逻辑op的输入或者输出）被映射到多个本地tensor上（比如逻辑op被放置的多个对应设备上）

### 3.1 指定每个Tensor和每个OP在分配的设备上的并行方式

设计的 SBP，是一个数学上的抽象层，指定一个全局 tensor 和对应的本地tensor之间的映射关系，包括：split(S)，broadcast(B)和partial-value(P)。比如图4展示了一个2x2的tensor被映射到2个本地tensor，其中有4种类型的 SBP mapping：
split(0), split(1), broadcast, partial-sum. 

![](./imgs/global-tensor-to-local-tensor.png)

从上图可见，split是指local tensor是沿着某个维度切分的，broadcast是指本地tensor是global tensor的拷贝。partial-value是指本地tensor和global tensor形状完全一样，而全局tensor是通过在所有本地tensor上执行逐元素规约操作（比如sum，max等）
得到的。

图5展示了在把逻辑图转换为物理图时，插入的 boxing op来完成数据搬运

![](./imgs/fig-5-data-movement-compiling.png)

### 3.2 Modeling Data Routing
模型的数据路由，主要讲自动插入通信的op来完成各种模型并行

### 3.3 Difference from GShard's Abstractions

partial-value 的好处是比立即reducing partital 的结果更高效. 有了 partial-value，oneflow 可以让系统选择最佳的时机来插入一个 boxing op（比如 reduce 或者 allreduce op）。
### 3.4 编程接口
编程接口的设计目标是让 op 的 API 和模型描述，在单卡和分布式版本下保持一样。而不同的分布式策略下，用户只需要设置一些tensor的placement和SBP的签名即可。

其逻辑图如下，就是两个 Matmul操作，只是涉及到数据并行和流水并行，而只需要在逻辑图上，设置每个Tensor的placement和SBP签名即可实现

![](./imgs/example-SBP-logical-graph.png)

```
import oneflow as flow
P0 = flow.placement("cuda", {0:[0,1]})
P1 = flow.placementt("cuda", {1:[0,1]})
a0_sbp=flow.sbp.split(0) # data parallel
b0_sbp=flow.sbp.broadcast # 
y0_sbp=flow.sbp.broadcast
b1_sbp=flow.sbp.split(1) # model parallel

A0=flow.randn(4,5,placement=P0,sbp=a0_sbp) # data parallel
B0=flow.randn(5,8,placement=P0,sbp=b0_sbp) 
Y0=flow.matmul(A0,B0)

Y0=Y0.to_global(placement=P1,sbp=y0_sbp) # 要 allgather吧？
B1=flow.randn(8,6,placement=P1,sbp=b1_sbp)
Y2=flow.matmul(Y0,B1)
```

上述的local tensor布局如下图，类似颜色代表是同一个global tensor的切分，相同颜色代表形状跟global tensor 一致。

![](./imgs/local-tensor-example.png)

## 4 运行时

## 问题
1. 依然是 SPMD？不会涉及到要把不同的函数分发到不同的设备上去
2. 

## 启发
1. SBP 之后，需要通信的地方，是可以自动推导出来的. 连 AllReduce 也是，这个也是因为目前 SBP 这种模式下特点
2. 
