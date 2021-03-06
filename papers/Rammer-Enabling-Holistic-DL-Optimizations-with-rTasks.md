想解决的问题本质：框架层的OP级别调度和硬件层面的Block Thread 两层调度之间的gap比较大（是不同层面的软件实现），导致效率不高。采用的是 whole program optimization。这个原则在 ML 里又用了一次。而 Google 的 IREE 也是 holistic(通盘) optimization

举例：上述提到的 inter op 与 intra op 调度是互相影响的。比如同一个算子的两种实现，一种较另一种消耗三倍资源，但只获得两杯加速。在 TensorFlow 这类单个算子独占整个硬件情况下，会选择更快实现。
但是两者协同调度下，选择资源“性价比”最高的实现而非“最快”往往是更优的选择。

为什么这个问题现在提出来解决是重要的？因为现在模型结构更复杂，有了 inter op 并发的需求。

## 前提

kernel 需要先 profiling 一下，要求实际执行时的性能表现是 deterministic 的。可以看看代码，这个 profiling 里关注的应该就是耗时和现存占用吧。


[开源的 NNFusion](https://github.com/microsoft/nnfusion) 是:
> a flexible and efficient DNN compiler, in which Rammer is integrated

核心原理就是加大并行，采用了三个方法：

1. 图里算子并发rTask（静态调度）
2. 设备抽象虚拟化，利用设备自身能力并发
3. 根据当前资源使用率做优化调度

感觉是比较通用，简单，因为没用到设备相关的intrinsic 指令，计算和存储叠加进行的技术，劣势是这种做法优化空间有限，而且需要设备并发编程模型api对外开放，类似cuda c。咋们stpu好做上述2吗？


又翻了下这个论文，思路很简单：编译期做好静态调度，目标是尽量加大并发。
具体做法：根据每个算子的profile结果，在资源许可的范围内统一考虑算子内部和算子间的调度
亮点：

1. 有算子间并行，GPU利用率较高
2. 调度开销小，运行时几乎没有调度了，是静态把执行计划拿过来执行
3. 调度策略和机制分离，可扩展

如果 STPU 上要用，需要：

1. 每个算子有两套实现：高效版（耗资源）和低效版（省资源）
2. 需要实现 virtual parallel device 接口

# 实现
Rammer 只用在论文里，实现叫 NNFusion，是一个端到端的 DL compiler。所有 Rammer 相关技术都实现为 NNFusion 里的一个 优化模块，叫做 BlockFusion。

## 论文和代码的对应关系
[Rammer -> NNFusion](src/nnfusion/engine/pass/graph/blockfusion/)

[RammerBase -> NNFusion](src/nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp)

[rOperator -> BlockCudaEmitter](src/nnfusion/engine/pass/graph/blockfusion/common.hpp)

[rProgram -> BlockExecutorProgram](src/nnfusion/engine/pass/graph/blockfusion/common.hpp)

[rTask -> BlockExecutorInstruction](src/nnfusion/engine/pass/graph/blockfusion/block_parallel_device.hpp)
[vDevice -> BlockParallelDevice](src/nnfusion/engine/pass/graph/blockfusion/block_parallel_device.hpp
)

## vDevice with two vEUs
下面是 PTB 的例子，可以在一个 PTB 里连续地运行多个rTasks，需要。

2个Matmul的 rTask 运行在 vEU0 上，4个 rTask的Relu运行在 vEU1 上。两者并行运行都结束后（各自运行一个barrier-rTask: vEU0 等待 4th rTask on vEU1 结束，vEU1 等待2th rTask on vEU0 结束），都执行 一个 conv2d 的 rTask
```
__global__ void vdevice_run() {
    if (Get_vEU_Id() == 0) { // vEU 0
       MatmulrTaskOp.compute_rtask(0); // 依次执行两个 Matmul？
       MatmulrTaskOp.compute_rtask(1);
       // wait the rTask on vEU 1 with order=3
       BarrierTask({<1, 3>}).compute_rtask();
       Conv2DrTaskOp.compute_rtask(0);
    } else if (Get_vEU_Id() == 1) { // vEU 1
       for(auto i : 4) {
           ReluTaskOp.compute_task(i);
       } 
       BarrierTask({0, 1}).compute_rtask(); // 这个 BarrierTask 语义如何实现？  while([vEU][rtask] != done) {}
       Conv2DrTaskOp.compute_rtask(1);
    }
}
```

## 

# 论文
##核心

把算子之间和算子内部一起考虑来做调度。跟硬件无关的优化

## 背景
由于 GPU 越来越强大，导致：

1. GPU使用率很低：2%~62%
2. op的调度开销很大：38%-65%的非Kernel Time

可以理解为是计算所需时间越来越少，但是op的调度开销没怎么变？

第一个截图咋理解呢？两个算子并发，前提得是资源有空余。右图里就4个大点的矩阵了，是说提高 kernel的资源使用率，空闲出 EU 来做 matmul？



## 挑战
### 1. 算子是不透明的函数，没有暴露细粒度的算子间并行
解法：使用 rTask-Operator(rOperator) 抽象

* 暴露细粒度算子内并行
* 是一组独立的，同质化任务
* rTask 是 单个EU 上最小的计算单元

### 2. 加速器（如GPU）没有暴露算子间调度的接口
解法：虚拟并发设备抽象层

* 暴露硬件的细粒度调度能力
* 让调度和硬件设备解耦
* 绕过硬件调度器

允许来自不同算子的多个任务运行在特定的一个vEU 上。一个 vEU 映射到一个物理EU上（那这不还是一一映射嘛？）

### 3. 细粒度的调度会导致更多调度的开销
幸好发现：DNN 计算是可预测的：

* 大部分 DNN 的 DFG 是编译期间决定的
* 算子的性能也是确定性的(跟上层业务无关？）

解决方案：在编译期就生成执行计划(rProgram)。把策略和调度机制分开。

* 在机制侧：1. 提供调度的接口，可以指定策略来产生一个执行计划。2.一个 profiler 来收集 策略。
* 在策略侧：开发了一个类似水波的调度策略。它能够结合算子内和算子间来做最大调度策略


在编译期就确定好调度策略的选择，确定好执行计划后，静态地映射到设备上。

好处：

* 避免运行时的调度开销
* 能结合算子内和算子间的



## Wavefront scheduling policy
* 每个 rOperator 有不同的kernel实现：最快的kernel，执行时间短，但是使用的资源大；节省资源的 kernel，执行性能差一些，但是使用的资源少。
* 把 DFG 根据 BFS 分成一拨拨的：在一波里的 operators 没有依赖，可以并行运行。每一波里，如果不会耗光资源，rammer就选择最快的kernel实现；反之选择资源高效的kernel实现
* 

## 疑问
1. Rammer 里训练 BERT，能让哪些算子并发呢？目前看来都是有前后依赖关系(Self Attention, MLP)，没有可以并发的把？

1. AlexNet 里没有可以并发的 OP 吧？除非做 op间流水？

## TODO
1. 看看 [Persistent Thread Block 论文](https://www.classes.cs.uchicago.edu/archive/2016/winter/32001-1/papers/AStudyofPersistentThreadsStyleGPUProgrammingforGPGPUWorkloads.pdf)
