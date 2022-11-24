1. 在解决的是什么问题？机器学习里几十亿参数量级的模型里，通信开销很大。对于特定网络拓扑和程序的通信模式下，自定义集合通信的算法，就行优化能缓解。但是正确、高效编写自定义算法难度很大。它提供 DSL 来简化书写自定义通信算法的难度，并且会使用编译器来优化他们到二进制格式，可以高效执行，是以运行时的解释器来执行的。
2. 为何成功，标志/准是什么？概念简单，实现起来好实现。实现的allreduce提速1.9倍，alltoall提速1.3倍
3. 在前人基础上的关键创新是什么？抽象出基本的几个概念后，用户用这些概念来描述数据路由就行，算法实现起来非常简单
4. 关键结果有哪些？不用担心data races、死锁等的情况，而且不用写cuda 就可以自定义实现通信算法
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？



## 1 介绍

NCCL 提供的库在某些系统配置和特定大小下很高效，但是很多情况下不是最优的。

GC3 是一个统一的框架，集提供了算法的灵活性，也有高性能。GC3里使用DSL来实现通信算法，使用编译器产生高效的可执行程序，有一个高效的gc3 执行程序的runtime。不用担心data races、死锁等的情况，而且不用写cuda 就可以自定义实现通信算法。而且，GC3自动检查实现是否是对已有的集合通信库的扩展。最后，提供的 API和NCCL兼容的，所以让已有的ML负载可以很容易切换到GC3，继承 NCCL 里对多种GPU的支持，对于还没不支持的算法，可以安全地 fall over 到 NCCL 的kernel里。

它有两个库：msccl-tools 和 msccl




## 2. GC3 Example
这种规格叫做 chunk-oriented。在GC3里，chunk 是通过它所在的rank 和 rank 内 buffer 里的下标来唯一表示的。然后这个chunk会被路由过去，然后拷贝。详情见 GC3 里的 DSL 解析，见第三节

## 3 GC3 DSL

### 3.1 Buffers and Chunks
GPU 显存在 GC3 里暴露为命名的 buffers，有三种类型：

1. input, 是保存输入数据的地方，
2. output和scratch 这两种未被初始化。

GC3 program 的作用是保证每个 GPU 上的 output buffer 都被集合通信的结果准确填充。scratch buffer 被当作临时的存储


Buffers 会被分成很多 chunks，代表了连续的元素，在整个 GC3 程序里都是均匀的大小。chunks 的数量是显示指定的。但是，程序粒度的 buffer size 是抽象的；大小只有当程序里具体的buffer被传递给gc3程序时，才会确定。chunks 有三种形态：

* 输入 chunks：在运行时初始化，唯一由input buffer 和 (rank, index) 来确定
* Reduction chunks: 通过一种逐个元素的规约(比如加法)来结合两种 chunks 。它是通过一堆被结合的输入 chunks 来表示的
* 未被初始化的 chunks： 在程序开始时，输出和 scratch buffer 里保存的就是未被初始化的 chunks

### 3.2 集合通信
没看懂里面提到的 postcondition 在哪里？这个约束了每个程序，可以用来做正确性验证

### 3.3 GC3 Operations
表格1里展示了用来操作chunks的所有操作。 chunk(rank, buffer, index, count=c) 返回C个连续的，被分配到 buffer 这个名字上的buffer，起始于index。count默认是1.为了正确性，GC3发现chunk没被初始化就被访问，会报错

使用copy 和 reduce 操作来在不同的 buffer 之间进行 chunks 的移动。c1.copy(rank2, buffer, index2) 是把 c1 所指的 chunks 移动到 (rank2, buffer, index) 这里。而 c1.reduce(c2)，可以规约出一个 inplace 修改 c1 的逐个元素让 c1和c2 进行 reduce的结果。copy 和 reduce 返回新生成的 chunks，这样可以做 fluently chaining copy 和 reduce 调用

### 6.1 点对点链接

**协议** NCCL 实现了三种协议：simple、LL128、LL，这三种是在延迟和带宽之间进行折衷。

Simple 有最高的带宽和延迟

LL 有最低的带宽和延迟

LL128 是居于之间的
## 5 调度的扩展，更多优化

## 其他
1. 关于 two-step 和 three step的图文解释，见我的keynote：“msccl alltoall通信相关”
## 问题
1. c.copy之后生成的 scratch buffer这种临时存储，命名是某个卡上唯一？不需要保证整个group粒度唯一？比如 two-step里的 c.copy(rankd2, f'copy_{n2}') # 每个主机上，会有8份拷贝给n2主机的数据
2. Buffer.Input，看起来是全局的，只不过靠它上面的rank和index来区分是当前这个rank里自己的，还是别人那里的？
