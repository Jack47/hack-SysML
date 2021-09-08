## 背景
### Parallel Programming 里通信模式有三种：

1. 同一台主机上，多个进程间共享内存
2. 分布式内存模型， MPI (Message Passing Interface)
3. Logical shared memory: Partitioned Global Address Space.

![](./imgs/supporting-programming-models-for-multip-petaflop-systems)

### 编程模型有
MPI, PGAS, CUDA, OpenACC, Cilk, Hadoop, MapReduce, Spark 等


### 编程模型上需要的通信库或运行时：

1. P2P
2. Collective Communication
3. Synchronization & Locks
4. I/O & File Systems

### 下面不同的硬件
Networking Technologies: InfiniBand, BlueGene, OmniPath

Multi/Many-core Architectures

Accelerators( NVIDIA and MIC)

## MPI
MPI 是给平行计算编程时使用的通信协议，支持 P2P 和 集合通信。是一个传递消息的应用程序编程接口，提供了协议和语义的规定，要求在任何实现下都遵守。MPI的目标是 高性能，可扩展，可移植。

### 概念
Communicator: 连接了 MPI session 里的多组进程。每个 communicator 给包含的进程们提供一个唯一独立的标识，把他们安排为有序的拓扑(rank i)。MPI 也有显示的 groups，但这些是为了组织和重新组织进程，在其他 communicator 制造之前。

Point-to-point basics: MPI_Send，可以给指定的进程发消息。
 
Collective Basics: 包含在一个进程组里，所有进程之间的通信。比如 MPI_Bcast，可以从一个进程获取数据发送给所有其他进程。MPI_Reduce 只好相反，从所有人那里拿数据，汇聚到一个节点上，成为一个结果。

Derived data types: MPI\_INT, MPI\_CHAR, MPI\_DOUBLE

MPI2 里定义了如下概念：

One-size communication: MPI\_Put, MPI\_Get, and MPI_Accumulate, 可以写入/读取从一个远程的内存里。

1. MPI processes can be collected into groups
2. Each group can have multiple colors (some times called context)–Group + color == communicator (it is like a name for the group)。所以 linklink 和 pytorch 的 distributed 可以共存，就是两个不同的 communicator
3. When an MPI application starts, the group of all processes is initially given a predefined name called  \MPI\_COMM\_WORLD

The same group can have many names, but simple programs do not have to worry about multiple names

A process is identified by a unique number within each communicator, called rank



Dynamic process management: 一个 MPI 进程可以参与到新的 MPI processes 创建，或建立连接。有三个主要接口：`MPI_Comm_spawn`, `MPI_Comm_accept`/`MPI_Comm_connect` and `MPI_Comm_join`

有 MPICH 实现，有 Open MPI 实现。
## MVAPICH2
是 BSD-licensed MPI 标准的实现，由俄勒冈州立大学开发。针对不同硬件，有不同版本： MVAPICH2(IB & RoCE), -X, GDR(GPU & IB), MIC, Virt

其特点：

1. 基于 RDMA 的跨主机 MPI p2p 通信，可以从/去 GPU 设备内存 (GPU-GPU, GPU-Host 和 Host-GPU)
2. 机器内 p2p 高效通信 (GPU-GPU, GPU-Host 和 Host-GPU)
3. 优化了 MPI collective 通信效率，从/去 GPU 设备内存
4. MPI Datatype 可以支持 p2p 和 collective 通信，去/从 GPU 设备内存
5. 利用 CUDA IPC(从 CUDA 4.1 开始) 在节点内多个 GPU 间通信的优势
6. 使用 CUDA Events 机制来提供高效的同步机制，提供给流水线化的去/从 GPU 设备内存发生的数据传输

## 问题
1. slurm 里如何支持 MVAPICH2

## 参考资料
1. https://wiki.mpich.org/mpich/index.php/PMI_v2_API，介绍了 PMI 和 MPI 如何配合，比如 PMI_Init 时，会返回 size, int *rank, int *appnum(MPI_COMM_WORLD) 等
2. https://anl.app.box.com/v/2019-06-21-basic-mpi： 介绍了 MPI里的基本概念，比如 MPI_COMM_WORLD，communicator,
