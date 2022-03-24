nccl-test 可用于测试 nccl 实际通信的性能。它提供这么三个指标：

1. ms 单位的平均操作时间: 这个对小块的传输有用，能展示出操作的额外开销(latency)。较大的大小下，就无法测出 latency 了
2. algorithm 带宽: size(S) / time(t)。这个对于 p2p 通信有用，比如 Send/Receive。它无法衡量集合操作的速度，因为跟gpu卡（rank）数量有关，随着数量变化，带宽增减
3. bus 带宽：能反映出硬件速度的瓶颈，比如 NVLink, PCIe, QPI 或网络。bus 带宽 **与 rank 数量无关**。这个数字会使用跟具体通信类型有关的公式来反映 GPU 间通信速度。用这个就可以比较硬件的 peak bandwidth

算法带宽 = (输入数据)/(传输数据) * (传输数据/执行时间) = 算法效率 * 总线带宽（algorithm bandwidth）

其中 in-place 是复用 buffer，原地置换算法。而 out-of-place 是单独有输入输出 buffer。取决于调用 nccl* 时 sendbuffer == recvbuffer
