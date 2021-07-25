## CUDA 编程模型
CUDA 自身采用的 SIMD(Single Instruction Multiple Data) 编程模型。线程是组织为块，然后不同的块被分配到不同的 SM(Streaming Multiprocessor)里。一个线程块的执行是通过把线程以 warp （一个为32个线程）的方式组织起来进行的。kernel发射的块可能远多于众多 SM 能一次处理的。所以硬件会在
一个块(block)结束后切换上下文，执行另外一块。所以程序员无法推断众多块是如何在 SM 上调度的。这种方式既是优势：简化程序员的心智负担，而对于无法完美符合 SIMD 编程模型的算法而言是一个限制。

TensorFlow 里只使用了一个计算流，多个拷贝的流。一些 kernel 可能选择了多计算流，但是维持了单个 stream 的语义。

为何不开启多流处理？因为 kernel 比较大，就已经能吃掉大部分 SM 了。开启效率提升不多，而且会比较复杂。

在一个 thread block 里是显示并行。在线程块间(kernels) 是隐式并行。

一个线程块之间的多个线程可以通过 `__shared__` 变量来共享数据

## GPU 上 Persistent Threads

本质：用软件调度取代硬件调度，应用于一些不太适合 SIMD 的场景。它是一种编程的方法，想解决一类问题


使用一个 Worker Queue 来存储要干的活，是某一类活，输入和输出都通过 kernel 启动时传递进去的。不能是不同类型的活，比如矩阵乘和图形渲染。但是对于可以算子融合的 Conv + BN 这类，是可以有 worker 做 Conv，判断到达一定条件后做 BN 操作。

### 什么情况下用它？
0. 需要在 Kernel 内部做全局的同步（不太理解）
1. 希望避免 Kernel 发射的延迟（复用kernel）
2. 是实现一种 producer-consumer 模型的方法：普通方式下需要生产者一个 kernel，消费者一个 kernel，如果两者在一个 cuda stream 里，就只能串行了。而如果用 Persistent Threads，可以把两个原来的 kernel 用 while 循环包起来，
3. 优化寄存器/片上存储的复用（比如 百度硅谷研究院的 Persistent NN）

来源：[Some specific reason to use persistent threads](https://forums.developer.nvidia.com/t/performance-of-persistent-thread-approach-on-new-gpu-architectures/43254/5)
PT 是解决上述问题的方法之一，绝不是唯一方法

### 适合哪些场景？
1. worker, producer 这类
2. 多个 kernel 之间需要全局同步？

### 不适合哪些场景？
1. 适合 SIMD 的，比如矩阵乘法？

### 优势

### 疑问
1. 

### 线程间如何同步？
1. 这里是同一个Kernel的不同线程块间(inter-thread block)的同步，使用 CUDA 提供的全局内存上的原子操作来协作，比如从队列里取数据：atomaticIncrement()

## torch.cuda.Stream

一个 CUDA stream 是一个线性执行序列，属于一个特定的设备，和其他stream独立。torch.cuda.Stream(device, priority)

synchronize(): 等待stream里的所有kernel 完成。是 cudaStreamSynchronize()的一个包装。

wait_stream(stream) : 和其他stream 同步

默认 GPU 操作是异步的。当调用一个 GPU上的函数，操作是排队到指定过的设备上，之后才有可能执行。

## 示例
```
// Persistent thread: Run until work is done, processing multiple work per thread
// rather than just one. Terminates when no more work is available

// count represents the number of data to be processed

__global__  void persistent(int* ahead, int* bhead, int count, float* a, float* b)
{
    int local_input_data_index, local_output_data_index;
    while ((local_input_data_index = read_and_increment(ahead)) <   count)
    {
        load_locally(a[local_input_data_index]);

        do_work_with_locally_loaded_data();

        int out_index = read_and_increment(bhead);

        write_result(b[out_index]);
    }
}

// Launch exactly enough threads to fill up machine (to achieve sufficient parallelism 
// and latency hiding)
persistent<<numBlocks,blockSize>>(ahead_addr, bhead_addr, total_count, A, B);
```
## 问题
1. 同一个 GPU 上，多个 stream 之间是并发的嘛？ 是的。 stream 的作用就是提供串行执行不同kernel的机会
2. 不同的 GPU 上，各自用自己的 stream，stream 之间是并发的嘛？ 是的
3. PyTorch 里 stream 和 TensorFlow 有何不同？ PyTorch 里可以新建 stream，让不同的 op 跑在不同的 stream 上
4. 我们的算子，有哪些可以用 Persistent Threads 来加速嘛？

## 其他参考资料：
1. [Persistent RNNs](https://hgpu.org/?p=16050) : 通过 Persistent Threads 来实现更高的计算吞吐，原因是通过 cache 这些权重(weights)，在多次推理过程中复用。本质是提高了寄存器/片上芯片存储的重用. [Github](https://github.com/baidu-research/persistent-rnn/blob/master/include/persistent_rnn.h) . [百度的论文](http://proceedings.mlr.press/v48/diamos16.pdf)
2. [CMU cs 15869 lecture 11 gp GPU](http://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/lectures/11_gpgpu.pdf)
3. [A minimum CUDA persistent thread example](https://gist.github.com/guozhou/b972bb42bbc5cba1f062)

