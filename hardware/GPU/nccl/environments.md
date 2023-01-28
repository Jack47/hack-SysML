为了有高效的 GPU 间传输性能，社区交付了 GPUDirect RDMA 技术。它提供了 NV GPU 和远端系统之间的直接通信，绕过了 CPU，避免了系统内存里数据拷贝，导致**性能提升很大**。

## NCCL_NET_GDR_LEVEL(formerly NCCL_IB_GDR_LEVEL)
since 2.3.4

它允许用户细粒度控制什么情况下，在 NIC 和 GPU 之间进行 GPU Direct RDMA 的操作。级别控制了 NIC 和 GPU 之间的最大距离。而取值是使用路径类型的字符串来指定拓扑类型

如果没有指定，NCCL 会尝试基于架构和运行环境最优地选择一个值

### Values accepted
LOC: Never use GPU Direct RDMA. (always disabled)

PIX: Use GPU Direct RDMA when GPU and NIC are on the same PCI switch

PXB: connected through PCI switches (potentially multiple hops)

PHB: on the **same NUMA node**. Traffic will go through the CPU

SYS: even across the SMP interconnect between NUMA nodes (e.g., QPI/UPI). (always enabled)

legacy：上述是0-4的取值，>5的会被认为是 SYS

## `NCCL_IB_PCI_RELAXED_ORDERING`
since 2.12

允许在 IB Verbs 传输上使用 Relaxed Ordering。它能极大帮助提升在虚拟环境下的 InfiniBand 网络的性能

### Values accepted
设置为:

* 2 能自动使用 Relaxed Ordering。
* 1 强制使用，如果不可用会导致失败
* 0 关闭

这个是 PCIe 的特性， Relaxed Ordering 可以允许在 PCIe 上的传输实物有灵活性。减少了 lane 上的传输次数，极大提升 **虚拟化网络** 里的IB 网络性能

## `NCCL_DEBUG_SUBSYS`
Since 2.3.4

这个变量让用户可以基于subsystem来过滤 `NCCL_DEBUG=INFO` 后的输出。在 NCCL debug log traces里，会包含逗号分隔的子系统

而给子系统以 ^ 开头，可以关闭这个子系统的日志
 
`NCC_DEBUG_SUBSYS=NIT,COLL,INIT,GRAPH,TUNING,ENV`

支持的子系统：

* INIT(initialization)
* COLL(collectives)
* P2P(peer-to-peer)
* SHM(shared memory)
* NET(network)
* GRAPH(topology detection and graph search)
* TUNING(algorithm/protocol tuning)
* ENV(environment settings)
* ALLOC(memory allocations)
* ALL (包含所有子系统)

[NCCL Performance Impact with PCIe Relaxed Ordering](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/nccl-performance-impact-with-pcie-relaxed-ordering/ba-p/3660825)



