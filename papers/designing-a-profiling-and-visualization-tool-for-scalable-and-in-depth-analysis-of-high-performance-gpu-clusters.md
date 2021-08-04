1. 在解决的是什么问题？ 提供统一和全局的MPI 和在 NVLink 网络硬件粒度相结合的实时分析、profiling 和 可视化工具。因为当时没有把网络硬件和 MPI 时间关联起来的方法
2. 为何成功，标志/准是什么？ 能给研究员、框架开发者、MPI 通信库开发者一些全局视角来优化
3. 在前人基础上的关键创新是什么？ 从上到下，把 CUDA/MPI/GPU内部 NVLINK 三个级别关联(correlation)起来
4. 关键结果有哪些？当发生端到端速度变慢后，能分析出到底是哪个层面，哪个粒度变慢了? 是应用程序的问题，还是 MPI 通信库，或者是硬件/机器层面的问题？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？启发大家，在 DNN 里需要一个全局视角，每个 iteration 粒度的 profiling 工具


利用了 OSU InfiniBand Network Analysis and Monitoring Tool (INAM)

## A. 本文主要贡献
* 设计了一个可以 profile 的 GPU 间通行库，通过 CUPTI 和 MPI_T 来收集 GPU 和 MPI 性能计数器
* 呈现了一个实时和低开销的 profile 工具来关联 MPI 层与网络层指标。这样就知道网络层当前指标主要是哪个 MPI 行为/操作触发的？
* 引入 MPI_T 基于事件的指标，是 MPI 库来做点对点和集合通信模式
* 添加了 PVARS 来准确量化 MPI 通信操作
* 在有 NVLink 的环境里实际运行并分析了效率
* 与 NVLink 类似，展示了给 MPI通信分析和分类流量和link 使用率

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
                                                  



