如何在CUDA C/C++ 里高效访问 Global Memory？

threads 是被分组到 thread blocks，它可以被分配到不同的 SM 里。当threads仔执行时，还有一个更细粒度的分组叫 warps。GPU上的 Multiprocessors 执行指令是以每个 warp 为单位，以  SIMD(Single Instruciton Multiple Data)方式进行。Warp size 一般是32个线程

## Global Memory Coalescing(合并）
线程分组到 warp 不仅对于计算有关，而且对于访问 global memory 也有关。设备上一个warp里的global memory 加载和存储会被合并到尽量少的 transactions 上去，来减小使用的 DRAM 带宽。

设备会把 warp 里的线程的global memory显存合并到几次 cache line 里。

## 总结
memory access liagnment 在近代的硬件上没那么重要了。但是 strided memory access 会影响性能，可以通过使用shared memory 来缓解

## 参考资料：
1. [How to Access Global Memory Efficiently in CUDA C/C++ Kernels(2013)](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
