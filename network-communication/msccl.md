msccl里allreduce实现有两种：

1.hierarchical_allreduce, 参考的这个[论文](https://proceedings.mlsys.org/paper/2019/file/9b8619251a19057cff70779273e95aa6-Paper.pdf)

2.[allreduce_ndv2](https://github.com/microsoft/msccl-tools/blob/main/examples/mscclang/allreduce_ndv2.py)



Msccl 诞生的目的是针对 azure 特定的环境和size，对通信进行优化，利用nccl 原语实现高效的通信