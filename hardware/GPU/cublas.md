
## 矩阵乘法 GEneral Matrix Multiply(gemm)
类似 cublas<T>gemm，支持浮点数和复数(Complex)的矩阵乘法

```
C = alpha*op(A)*op(B) + beta*C
```
其中的矩阵都是 column 为主的格式，维度是 op(A): m x k, op(B): k x n ，然后 C 是 m x n. 对于 A：

transa 可以是 `CUBLAS_OP_N`: 不转置，`CUBLAS_OP_T` 和 `CUBLAS_OP_C`

```
cublasStatus_t cublasSgemmEx(
    cublasHandle_t handle, # cuBLAS library context.
    cublasOperation_t transa, # 是否、做什么样的转置
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, # 乘法的标量系数
    const void *A,
    cudaDataType_t Atype, # 枚举类型，表明矩阵 A 的数据类型: CUDA_R_16BF, CUDA_R_8I, CUDA_R_16F, CUDA_R_32F 等
    int lda, # 存储矩阵A的二维数组的第一维
    const void *B,
    cudaDataType_t Btype,
    int ldb,
    const float *beta, # 矩阵C的系数
    void *C,
    cudaDataType_t Ctype,
    int ldc
)
```
