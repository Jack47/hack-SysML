## Shared Memory Note: 

需要注意分配的内存，如果分配太多，可能出现 kernel 里无法执行任何指令的情况，此时 cuda-GDB 可以执行，但是即使 printf 也无法运行
```
<<<blockNum, threadsPerBlock, sharedMemSize>>>

```

## What is BLAS?
Basic Linear Algebra Subprograms. It defines a set of common functions we would want to apply to **scalars**, **vectors**, and **matrices**.

当时是在 FORTRAN 上写/用的，后来也follow 这个规范

名字晦涩

### 特点：
1. 矩阵是以列优先的方式来索引的
2. cuBLAS 里的数组是一维的，所以可以定义一个宏来做索引

### 三类运算
1. 标量和向量运算
2. 矩阵和向量运算. 标量可以换算成向量，向量可以变成矩阵：1xN 的
3. 矩阵和矩阵运算. 

一般都会写一个 C++ 的模版，套在 cuBLAS 上

cuBLAS 主要适用于通过多 stream 来批量执行多个 kernel 的情况，比如在密集矩阵上进行多个小矩阵的乘法

每个 cuBLAS 里对应 CUDA C 里的 4 种数据类型: S, D(double), C(complex), Z, H(Half)

举例：

cublasSgemm: single general matrix multiplication

cublasHgemm: same as before except half precision

## 如何使用？
1.  #include cublas_v2.h
2. link the library using `-lcublas`

## 举例

### 向量乘矩阵

numpy.dot(alpha, B) # 点乘

<==> 

cublas<T>dot(alpha, B)

x*y # 星乘

### 矩阵乘

numpy.multiply(alpha, B) # 点乘

numpy.multiply(A, B) # 点乘

<==> 

cublas<T>gem(alpha, B)


normalize or  scale:

## 下标索引

#define IDX2C(i, j, ld) ((j)*ld + i)

In column major storage "ld" is the number of rows.

## 问题：
1. 矩阵和矩阵之间，只有一种矩阵乘法？
2. 向量和矩阵之间，只有一种 elementwise 的乘法？
