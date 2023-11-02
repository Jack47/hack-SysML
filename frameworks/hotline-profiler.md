它无需修改  DNN lib 或者使用 vendor 提供的工具，没有引入测量时候的额外开销。

## 1 介绍

看看这几篇介绍已有的 ML profile 工具的论文：

dPRO: A Generic Performance Diagnosis and Optimization Toolkit for Expediting Distributed DNN Training( 2022)

Google: Perfetto Trace Processor 2022. TensorFlow Profiler(2022b)


## 疑问
1. 数据怎么保存的，profile 10 分钟，是否会产生大量数据
2. 额外开销是多大？
3. 劣势
