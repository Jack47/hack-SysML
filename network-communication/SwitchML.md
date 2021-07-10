ML 程序里，每个 iteration 都需要在主机间进行梯度的 all-reduce 操作，传输的数据多大 100MB~1GB/s

主要挑战：
1. 交换机上计算和存储的空间有限
2. 不支持浮点数
3. 会有丢包情况

设计：
1. 结合 switch-host 的架构。意思是switch 有自己的host?
2. 基于池子(Pool) 的流式聚合
3. 量化整型操作
4. 失败后可恢复的协议
5. 在交换机内部的 RDMA 实现

交换机上可编程 6.5Tbps的通信带宽



1. 结合 switch-host 的架构。意思是switch 有自己的host? 
![](./imgs/switch-host-arch.jpg)
不是，看图是每个host 上的 worker 有量化和容错机制


## 实现细节
1. 交换机上的程序是给 Intel Tofino 写的，P4 语言
2. host 上有 C++ 库，提供 all-reduce 类似的 API
3. 跟 ML 框架如 PyTorch，TensorFlow 等集成了
4. 有 benchmark，可以测聚合的效率

在 100Gbps的网络里，对吞吐有 2.27x 提升。GPU 越快，效果越好，因为减少了计算/通信的占比

SwitchML 的性能(聚合消耗的时间)不取决于 worker 的数量。而传统的 NCCL，在机器数量很多时（具体多少？），会有效率下降。这是因为

## 原理
交换机上面跑 P4 写的程序，还有一个 controller 是跑在 Host 上，给交换机上的程序发控制消息

### Switch Controller
使用 Barefoot Runtime Interface(BRI) 来在运行时给交换机编程。Controller 会与 end-hosts 通过 gRPC
协议建立连接，配置好一个训练任务（是一个进行 allreduce 操作的 worker 序列）。也有 CLI 可以配置和读取运行时的计数器

交换机需要知道哪些流量是需要做 all-reduce 的，所以 controller 通过 ports.yml 来读取连接到交换机的端口。是用端口和 网线来区分的：
* speed : 10G, 25G, 40G, 50G, 100G
* fec: none, fc, rs
* autoneg: default, enable, disable
* mac: 连到端口的网卡的 mac 地址

启动： python switchml.py

BFRuntime server 是交换机的驱动程序

### SwitchML P4 program
![](https://www.intel.com/content/dam/www/public/us/en/images/product/RWD/tofino-chip-rwd.png.rendition.intel.web.225.225.png)

用专门给使用了 [Intel 的 Tofino](https://www.intel.com/content/www/us/en/products/network-io/programmable-ethernet-switch/tofino-series/tofino.html) 这类可编程 AISC 交换机 P4 设备数据面编程用的语言

[Open_Tofino 项目](https://github.com/barefootnetworks/Open-Tofino)

Tofino 有 64x100G 端口，最大带宽： 6.4Tbps。Frame Processing Rate：4.8B pps, 与 CPU 之间是 PCIe 的接口


参考：

1. [Cheetah is a system that optimizes queries using programmable switches.](https://github.com/harvard-cns/cheetah-release)
2. [Programmable Ethernet Switch: Tofino Series](https://www.intel.com/content/www/us/en/products/network-io/programmable-ethernet-switch/tofino-series/tofino.html)

## 限制
1. allreduce 操作必须是 sum
2. 数据类型必须是 float 或 int32
3. 每个主机产出一个 tensor，或者说每个主机用一个 GPU

另外，要编译上述的 P4 程序，需要 P4Studio 工具套装，还需要签署 NDA 和 SLA，才能获得文档和 P4 系列套件。P4编译器编译出来的结果，或者其他 P4Studio 工具的产出物，比如 context.json，bfrt.json，logs等，也不能发布到外面

Intel 搞的网卡交换机上的 ASIC。

## 疑问
1. 并不能跨switch 吧？因为池子是维护在一个交换机上的
2. 为啥worker数量增加，性能不变？worker 数量增加后，不是得等所有人到了，才能 reduce ？ 或许可以 stream 计算，来一个计算一个。switch 里的 Pool 只取决于要合并的 tensor 数量
