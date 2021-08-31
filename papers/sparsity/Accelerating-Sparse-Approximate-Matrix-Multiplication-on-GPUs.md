1. 在解决的是什么问题？ GPU 上实现高效的 near-sparsity
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？优化了 sparse approximation 算法
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 概念
根据矩阵里0的数量，可以分为三类矩阵：

Dense Matrix: O(N^2)

Sparse Matrix: O(N)

Near-sparse Matrix: 在上述两者之间

近似算法可以用来加速 near-sparse GEMM。例如，跳过里面足够小的元素。所以已经有一个 Sparse Approximate Matrix Multiply 算法

可以控制近似的程度（精度）
