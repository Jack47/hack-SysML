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

## TODO
1. 看看 [Persistent Thread Block 论文](https://www.classes.cs.uchicago.edu/archive/2016/winter/32001-1/papers/AStudyofPersistentThreadsStyleGPUProgrammingforGPGPUWorkloads.pdf)
