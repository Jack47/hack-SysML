## 是什么？
跟 CUDA Core 类似，是单独的运算单元，只能用来计算矩阵乘累加

在 cuDNN v7.6.3(V100) 和之后，卷积维度会自动被填充(auto padding)，来方便利用到 Tensor Core
## 如何用？
三种 API：

1. CUDA Warp Matrix Multiply Accumulation: wmma
2. CUTLASS
3. cuBlass

## 一些规则
1. 只能给GEMM类操作，而且是 math limited 的情况做加速，反之 memory limited 算子加速不了
2. 对于线性层：需要 input size, output size, batch size 是 8（fp16）或者16(int8)倍；对于卷积层：输入或输出的 channels 也是同样规则
3. 想要 TC GEMM 效率高：上述维度是 64/128/256 的倍数。如果 tiles 总数比较小，保证是 SM 数量的倍数
4. 如果 GEMM 的任意维度是 128 或更小，那 OP 就很可能是 bandwidth limited

## 如何知道启用了 Tensor Core？
1. nvprof，然后检查函数名里有这种模式：`[i|s|h][some numbers]`: `volta_h884gemm` `turing_fp16_s1688cudnn_fp16`. 但可能不充分
2. nsight system等里面会有两列指标：TC eligible, TC used


## 问题
1. 上述条件，是 input size or outputsize，还是乘积，或者是都需要是8的倍数？
## 参考资料

1. [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
