1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 主要贡献：
1. DSL语言：Tensor Comprehensions
2. 多面体 JIT  编译器，可以把 DL 里数学话描述的 DL DAG 转化为 CUDA kernel 函数，有内存管理、同步和优化方法，如op 融合，特定尺寸的特殊处理
3. autotuner 操作的编译cache

## 挑战
要想在 GPU 上高效实现，而又对研究员友好，需要一个高效的计算图引擎，解决如下两个挑战：

1. 抽象不仅提高程序员的效率，而且让 compiler 和支撑它的执行环境消除跟目标环境无关的顾虑。能自动搜索优化空间。the system must be able to "abstraction without regret".
2. 选择恰当的中间表达式和优化算法，处理好深度并发和内存层次结构，同时用好硬件特性如vector  指令和特定用途的内存。
## 问题
1. 论文里提到可以和 Cafee2 和 PyTorch 结合，通过 ATen 异步 tensor lib。这个咋做到的，为什么要这么搞？只是借 PyTorch 客？
2. Tensor Comprehension 发布 0.1.1 版本后，咋就 archived 了？
