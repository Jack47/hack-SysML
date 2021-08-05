1. 在解决的是什么问题？ 提供统一和全局的MPI 和在 NVLink 网络硬件粒度相结合的实时分析、profiling 和 可视化工具。因为当时没有把网络硬件和 MPI 时间关联起来的方法
2. 为何成功，标志/准是什么？ 能给研究员、框架开发者、MPI 通信库开发者一些全局视角来优化
3. 在前人基础上的关键创新是什么？ 从上到下，把 CUDA/MPI/GPU内部 NVLINK 三个级别关联(correlation)起来
4. 关键结果有哪些？当发生端到端速度变慢后，能分析出到底是哪个层面，哪个粒度变慢了? 是应用程序的问题，还是 MPI 通信库，或者是硬件/机器层面的问题？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？启发大家，在 DNN 里需要一个全局视角，每个 iteration 粒度的 profiling 工具


注意：本文讨论的是 MPI，与 NV 自己实现的 NCCL 不一样

利用了 OSU InfiniBand Network Analysis and Monitoring Tool (INAM)

## A. 本文主要贡献
* 设计了一个可以 profile 的 GPU 间通行库，通过 CUPTI 和 MPI_T 来收集 GPU 和 MPI 性能计数器
* 呈现了一个实时和低开销的 profile 工具来关联 MPI 层与网络层指标。这样就知道网络层当前指标主要是哪个 MPI 行为/操作触发的？
* 引入 MPI_T 基于事件的指标，是 MPI 库来做点对点和集合通信模式
* 添加了 PVARS 来准确量化 MPI 通信操作
* 在有 NVLink 的环境里实际运行并分析了效率
* 与 NVLink 类似，展示了给 MPI通信分析和分类流量和link 使用率

本质：获得单个节点内 NVLink 的链接拓扑：各自 id，然后和 MPI 的某个具体操作关联起来（时间和 uuid）

### 底层的 GPU Profiling 工具接口
CUDA Profiling Tools Interface( CUPTI) 提供了 tracing 和 profiling 目标 CUDA 程序的 API。有如下五类：

Activity APIs ： 可以异步收集 GPU 或程序的 CPU 活动，比如用来发现 NVLink 和 PCIe 的拓扑
Callback APIs
Event API
Metric API
Profiling API:

### MPI Tools Information Interface
提供标准机制来让 MPI 工具开发者来完成 inspect 和 tweak 众多内部设置的方法。提供两类方法：
1. PVAR(performance vars): peak under the hood of the MPI 库来决定状态
2. CVAR(Control vars)：可以改变 MPI 库里的参数

## 3. 设计高性能、低开销、可扩展的 GPU profiling 工具

![](./imgs/infiniband-network-analysis-and-monitoring-tool-arch)

数据结构：

### Intra_node_topoo
记录的是一根 NVLink 的信息，它链接了两个 GPU。这个信息是静态的

这个是某个主机级别的
Id, Node_name（物理主机)，Physical\_link\_count（不是有几根 NVLink,而是序号？），Link\_capacity（带宽？），Source, Source\_id， destination, destination\_id

### NVLink_metrics
记录 CUPTI 里获取到的 NVLink 指标：其中 dest\_global\_rank 这类是当前任务里全局唯一的次序 id。相当于关注的是当前任务粒度的

id,Link\_id, **Node\_name**, Source\_name, Source\_port, Source\_id, Dest\_name, Dest\_port, Dest\_id, Added\_on, Source\_local\_rank, **source\_global\_rank**(这个是 MPI 里的，如何拿到呢？可能是初始化时传递给了 CUDA？), dest\_local\_rank, **Dest\_global\_rank**, Data\_unit, Data\_recv, Data\_sent, Data\_recv\_rate, Data\_sent\_rate

### PVAR_table
记录的某次 MPI 粒度的数据？

Id, jobid, **Node\_name**, start\_time, end\_time, bytes\_recv, bytes\_sent, PVAR\_name, algorithm(gather, scatter or all-reduce?), **source\_rank**, **dest\_rank**, added_on

## 问题
1. 标题里的 scalable 体现在哪里？ 计算了一下一个节点上 GPU数量增加时 query gpu coutner 的 overhead，发现跟 GPU 数量成线性关系，就说明可扩展（哭）。而不同主机间是并行的，所以跟节点个数无关
2. 如何达到 low overhead ? 如何衡量引入的 overhead？
3. NVLink\_metrics 里的 Data\_recv 是当前任务从启动到现在，总共的接收数据？
4. 为什么 V 里最后部分说 Resnet50 里，batchsize 越大，通信利用率越高？传输的数据并没有增加，而且由于计算增加了，导致两次同步之间的时间会增加呢 =》 是说同步的次数变少了，所以效率增加了

## 启发
1. 我们可以精确知道模型训练时，哪个时刻进行了什么类型的 MPI 通信，耗时多久，用了哪个NVLink发送了多少数据，还可以知道背后有哪些 NVLink 参与，各自的带宽是多少？

## 参考资料
1. [MPI Micro-benchmarks ](http://mvapich.cse.ohio-state.edu/benchmarks)
2. [2018 年 pdf 介绍](https://www.hpcadvisorycouncil.com/events/2018/australia-conference/pdf/WednesdayAug29/HSubramoni_OSU_IntlDialogSysts_Wed082918.pdf)
