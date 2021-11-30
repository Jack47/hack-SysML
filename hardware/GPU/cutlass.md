cutlass: CUDA **Templates** for Linear Algebra Subroutines

cutlass vs cuBLAS

cuBLAS 文档丰富，速度比 cuTLASS 快。线上环境一般用 cuBLAS。但是 cuBLAS 没有开源，不够完整。有些情况下想扩展它，却不太可能实现。cuBLAS 内部如何工作，是由纯 CUDA 实现还是 PTXAS 实现并不知道

而 cuTLASS 社区会分享算法的解释和完整的代码实现。[参考自这里](https://github.com/NVIDIA/cutlass/issues/109)

## 问题
1. [CuBLASLt 底层用的是 CUTLASS kernel](https://github.com/NVIDIA/cutlass/issues/109#issuecomment-701007624)? 这不就互相使用了么？
