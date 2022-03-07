[ProcessGroupNCCL.cpp](https://github.com/pytorch/pytorch/blob//torch/lib/c10d/ProcessGroupNCCL.cpp#L812)

## ProcessGroupNCCL
当前类里提供的所有 NCCL 函数(p2p, collective) 都是异步函数。每个NCCL 调用都是在单独的 CUDA stream 而非当前的 CUDA stream。目的是为了
提高潜在的并发和更好性能。由此，调用者需要负责确保自己代码根据需要来和本类里的 NCCL 操作同步。

方法：


## syncStreams
让输入的 ncclStreams 等待当前 stream。NCCL 通信跑在 ncclStreams，但输入的 tensor 却分配在不同的流上（比如当前流）。在 ncclStream 上的通信
在挂起在当前流上 input tensor 上的op执行结束前，是不能开始的。否则这两个流上的操作可能会有并发读/写同一个 tensor 的情况。

上述同步还不够。我们还需要确保输入 tensor 在 ncclStream 上结束前不能被释放掉。可以用通过调用 `c10::cuda::CUDACachingAllocator::recordStream()` 实现。
它会记住使用的流，在之上创建一个 event，当 GC 要释放输入流时，延迟 GC 直到事件结束。

```
void syncStreams(devices, ncclEvents, ncclStreams)
```

## collective
因为是发生在一个进程里，而一个进程可能最多管了8个 GPU。所以有多设备，每个设备又会有多个流。但每个设备上，ncclStream 只会有一个。

```
// 1. NCCL streams 等待输入tensor allocation streams
syncStreams(devices, ncclEvents_[key], ncclStreams_[key])

// 2. 跟 syncStreams 里提到的，需要让 CUDACachingAllocator知道当前输入、出 tensor 上挂了一个 ncclStream，防止 ncclStream 还没用完，这个 tensor 就被 GC 了
c10::cuda::CUDACachingAllocator::recordStream(inputs[i].storage().data_ptr(), ncclStream)

// 3. 真正执行 fn
fn(inputs[i], outputs[i], ncclComms[i]->getNcclComm, ncclStream)

```

## allreduce
给外层暴露的 allreduce，只需要两个参数：需要操作的 tensor 以及用什么运算（sum）
```
std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::allreduce(std::vector<at::Tensor>& tensors, const AllreduceOptions& opts)

return collective(tensors, tensors, [&](input, output, comm, stream){}); // 可见通信主要是在某个stream上，用 communicator
```



