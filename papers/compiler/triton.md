1. 在解决的是什么问题？使用 DL 加速器 vendor 的库来加速算子太难写了，门槛太高
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？ 基于 tile(**静态**shape的多维子数组)来提供了语言和编译器，有一堆编译器优化算子来构建高效的 gpu 代码
4. 关键结果有哪些？和手工优化过的vendor 库如 cuBLAS/cuDNN 有同等加速水平
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 摘要

我们的方法包括：

1. C-based 语言和一个 LLVM-based IR 来表达以参数化 tile 变量 tensor programs
2. 几个创新的 tile 级别优化passes，可以把这些程序编译为高效的 GPU 代码。

我们实验证明了 Triton 可以用来构建可移植的矩阵和卷积kernel 实现，能够达到手工优化的 vendor lib (cuBLAS/cuDNN)一样的速度，也可以高效实现最近的研究新思路比如 shift conv

## 介绍

vendor libs 由于只支持一部分有限的tensor操作集合，让 sys ml 专家自己实现创新的原语。

这也导致出现了很多给 DNN 的 DSL 语言，基于多面体（Tensor Comprehensions) 或者循环合成的技术（Halide，TVM和 PlaidML）。

利用多面体模型来分析循环嵌套的依赖。这些工作都是自底向上。Fireiron 和 Stripe 使用嵌套的多面体结构来建模 tensor programs 以自顶向下的方法。

TensorIR 聚焦在自动化 tensorization 的进程来产生多平台上优化的代码，而无须人工干预。

AutoTVM 引入基于学习的方法通过一个学到的cost model 和模版指导的 search 来优化tensor 程序

Ansor 使用层次化搜索空间来提高自动调度

但这些

## 和 TensorIR 的对比

triton 功能上差不多是 TensorIR 的一个子集：设计了一个 tile-level programming interface，然后自动做了一些 schedule: 比如 ti.dot 背后的 tensorize

