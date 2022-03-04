本文是这篇[Overlap data transfers cuda](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
## CUDA Streams
CUDA 里，一个 stream 是在 GPU 上执行的顺序是按照主机上发射顺序来执行的一系列操作。一个流里的操作是保证按照预定顺序执行，但不同流里的操作可以重叠，而且有可能的话，会并发执行

## default stream

所有的设备操作（kernel和数据传输）在CUDA中属于某个流。如果没有指定流，会使用默认流(也叫 null stream)。它跟其他流不一样，因为是在设备上**同步执行**(synchronizing)的流：只有当所有在它之前发射的任意流（同一个设备）上的操作都完成后，默认流中的操作才会开始。
而且默认流中的这个操作必须完成，其他后续操作（同一个设备里的任意流）才会开始。

2015年发布的CUDA 7 里，引入了新的选项来让主机上每个线程粒度使用单独的默认流，把per-thread default stream 当作常规的流(不会和其他流里的操作同步)。参看[GPU Pro Tip: CUDA 7 Streams Simplify Concurrency](https://developer.nvidia.com/blog/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)

```
cudaMemcpy(d_a, a, numBytes, cudaMemcpyHostToDevice);
increment<<<1,N>>>(d_a)
cudaMemcpy(a, d_a, numBytes, cudaMemcpyDeviceToHost);
```
上述代码，从设备侧来看，三个操作都是发射到同一个（默认）流，所以会按照发射顺序来执行。从主机侧看来，数据传输是阻塞或说同步的传输，而 kernel 发射是异步的。所以第一行的 host到设备的数据传输是同步的，CPU 线程会在 h2d 传输结束后再执行到第二行。
当kernel发射后，CPU线程会执行到第三行，但是这个传输因为设备侧要保证执行顺序(default stream)，所以不会开始执行。

可以看到上述提到的 kernel 发射这个操作是异步的。

## Non-default stream
非默认流需要显示创建。

为了在非默认流里:

1. 发射数据传输，需要使用 cudaMemcpyAsync(stream1) 函数（是说必须？），它在主机侧是非阻塞的，控制流会在传输过程发射后立即回到主机。
2. 进行 kernel 计算，需要指定 stream 标识符里的第四个执行参数配置（第三个是分配共享设备显存）：`increment<<<1, N, 0, non_default_stream1>>>(d_a)`

cudaStreamCreate()

## Synchronization with streams
由于非默认流里的所有操作都是非阻塞的（从host侧代码看），所以会需要需要让host code 和流里的操作同步的情况。有几个方法：

1. cudaDeviceSynchronize()：粒度最大，会阻塞到设备上之前所有发射的操作都完成。这个粒度非常大，所以对性能影响很大
2. cudaStreamSynchronize(stream1)：等待单个指定的 stream1 里所有操作执行完 
3. cudaStreamQuery(stream1) 与2的差别是它不会阻塞 host 侧运行，只是去查询，而不是等待所有操作执行完 
4. cudaEventSynchronize(event), cudaEventQuery(event) 跟2，3类似。
5. cudaStreamWaitEvent 

## 数据传输和kernel计算重叠进行
所以想要达到重叠进行，就不能用默认流

多个 stream 之间 synchronizes 是指串行还是？

## 问题
AllReduce 内部会全部分段还是一整个传输呢

NCCL 里的基本概念：ring, CTA, communicators
TODO：
看看里面的链接

