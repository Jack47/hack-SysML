传统的从 CPU 给GPU喂数据方法是 PCI Express，它连接了 GPUs 和 CPUs。这种在桌主机和笔记本里常见
![](https://blogs.nvidia.com/wp-content/uploads/2014/11/nvlink.png)
如图，NVLink 即连接了 GPU 和 CPU，又做了 GPU 间的连接。看起来 NVLink 更大价值是用于同主机 GPU 间通信加速

![](https://developer-blogs.nvidia.com/wp-content/uploads/2014/11/nvlink_sort_perf.png)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2014/11/nvlink_configurability.png)

![](https://www.nvidia.com/content/dam/en-zz/Solutions/technologies/multi-instance-gpu/nvidia-nvlink-bridge-a100-pcie-diagram-dlt.svg)

从下图看，是GPU间点对点都有专用的4条线路？

![](https://www.nvidia.com/content/dam/en-zz/Solutions/technologies/multi-instance-gpu/nvidia-nvlink-a100-diagram-dlt.svg)

对于8卡，是这样的：
![](../../imgs/nvlink-p2p-gpu-interconnection.png)

从图中看到并不是**完全**点对点，每个 GPU [最多支持6条线路(links)](https://en.wikipedia.org/wiki/NVLink)
