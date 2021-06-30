## 问题
1. dataloder 里的 workers_num 数量的 worker，如何把数据传回给主进程？通过 input queue 和 output queue
2. 为什么有时候出现文件描述符超限制？因为共享策略选择的 file_descriptor，shm_open(fd)，这样传递的 tensors 数量多了就会导致超限制。


## 最佳实践

torch.multiprocessing 是 Python版 multiprocessing 的替代品。支持同样的操作，但是扩展了一下：所有通过 multiprocessing.Queue 发送的 tensors，其数据都会被**移动**（移动还是拷贝？）到共享内存，将会只发送句柄给对方进程。（效率会很高，避免内存重复拷贝）。

问题：那相当于 torch 里的实现才这样？

当一个 tensor 被发给其他进程，Tensor 数据是被共享的。当 torch.Tensor.grad 不是 None，也被共享。但是当发送给对方进程时 torch.Tensor.grad 是None，而之后才创建的 grad，是进程特定的，**不会自动在所有进程里共享**，不像 tensor里的 data。

### 多进程里的 cuda
cuda 运行时不支持 fork 这类的函数。

跟 CPU tensor 不同，发送进程被要求一直保持原始的tensor，直到接收进程获得一份 tensor **拷贝**。虽然这个机制在背后已经被实现了，单依然要遵守最佳实践。比如发送进程在接收进程拥有tensor引用过程中要一直活着，如果消费进程因为fatal 信号而推出，refcounting 也没法帮忙。

### 最佳实践
#### 避死锁
死锁主要原因是**后台线程**。当有任何线程拿到了锁或迎入一个 module，此时 fork 掉用了，那么子进程的数据状态很有可能是坏的(锁是无法在 fork 下生效嘛)，会死锁或其他方式失败。即时你没有掉用线程，Python 内置的库可能会。比如 multiprocessing.Queue 的实现非常复杂，会 spawn 多个线程来序列化，发送和接收数据，会导致前面的死锁问题。如果是这种情况，尽量使用 multiprocessing.queues.SimpleQueue, 背后没有额外线程。

#### 复用通过 Queue 来传递的 buffers
每次掉用 multiprocessing.Queue，需要移动到共享内存。如果已经共享了，就不用操作。否则会发生额外的内存拷贝，会降慢速度。即时你有一个进程池，发送数据给单个，然后把 buffer 返回 -- 这也是几乎免费的，会在发送下一个 batch 时避免拷贝。

## CPU tensor 共享的机制
不适用于 GPU tensor，因为 CUDA tensors 只能使用cuda api。

### file_descriptor
receiver will also cache the file descriptor and mmap，获得一个在数据上的共享视图。
### file_system
用文件名称而非具柄来唯一标识 共享的内存区域。

这种劣势是可能进程没了，但是文件还在系统里，导致内存泄漏。只能重启系统或手工释放




