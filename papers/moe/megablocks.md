1. 在解决的是什么问题？单个 GPU 上有多个 Expert 的场景下，如何解决一个 GPU 里计算专家复杂不均衡问题
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？把 HPC 里的 Variable Sized Grouped GEMM 实现用在了 MoE 里：核心是将多个 GEMM 操作分解成很多小矩阵块的 GEMM 操作，这样他们同时计算来完成大矩阵的 Variable Sized Batched GEMM
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？没解决 GPU 间不同专家计算不均衡的问题
6. 这个工作可能有什么深远的影响？单 GPU 多 expert 情况，在某些场景比如推理下很适合


激进地抛弃了 expert capacity（那怎么做的呢？），而且没有 zero-padding 开销。


关键技术是 group gemm：当每个卡上有多个 experts 时，可以把多个(可能比较小） local gemms 合并到一个 kernel 里，提高了 GPU 利用率和性能。[CUTLASS 2.8 里引入的](https://github.com/fanshiqing/grouped_gemm)

megatron-lm 里已经支持了 grouped gemm
