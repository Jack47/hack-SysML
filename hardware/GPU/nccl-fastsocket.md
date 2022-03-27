## NCCL Fast Socket
1. 使用多个网络流(network flows) 来最大化吞吐。借助了 NCCL 内置的 multi-stream ，让多个通信请求可以重叠。这个可能对我们有启发？可以看看他咋实现的
2. 动态在多个network flows 之间进行负载均衡。straggler network flows 就不会显著降低整个 NCCL 的集合通信操作。这个还是针对云上的，两个点之间有多个路径可达的情况。这样不同path的带宽不一样。提的都是100G以太网络里做实验


## Reduction Server

算法带宽 = (输入数据)/(传输数据) * (传输数据/执行时间) = 算法效率 * 总线带宽（algorithm bandwidth）

提高总线带宽的方法：提高硬件带宽，优化网络栈

提高算法效率：降低数据传输量：

* 最优的 Allreduce 算法效率上界：n/2(n-1) ~= 1/2
* Reduction Server 可提升到1(感觉它没算拿回来的, 说明大家都是按照要发送的量来算)

## 问题
1. TCP 通信和 NCCL 不是一个东西？


## 启发
1. 以太网 Socket 场景下，对于小消息，他们是跳过了一些步骤来发送，核心是提高延时。对于大消息，主要是做好负载均衡。这个思路是否可以借鉴？
2. Reduction Server，是不是也是给云上用的呢？在公司这种土豪网络方案下，是可以的嘛？

## 问题
allReduce 的朴素实现：

* reduce-scatter: 为啥是 (n-1)/n
* all gather

## 参考资料：
1. [Google Cloud 上的介绍](https://cloud.google.com/blog/products/ai-machine-learning/how-to-optimize-google-cloud-for-deep-learning-training)
2. [LanChen在 DataFun 的分享](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247566835&idx=1&sn=0382cccd76b6b2509907be1b703fe7c2&chksm=fbd6779fcca1fe895ad0d361b31ba61a78061337124cfe2f3f32803fdc4ea9a12e4af359249a&mpshare=1&scene=1&srcid=03174sljwCpIPN6pBSmua7S8&sharer_sharetime=1648131228367&sharer_shareid=2341dd0b8f3d7cdff1b600532fc15b3a&version=4.0.2.90474&platform=mac#rd)
3. [NCCL FastSocket](https://github.com/google/nccl-fastsocket)
