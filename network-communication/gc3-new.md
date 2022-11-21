
## 2. GC3 Example
这种规格叫做 chunk-oriented。在GC3里，chunk 是通过它所在的rank 和 rank 内 buffer 里的下标来唯一表示的。然后这个chunk会被路由过去，然后拷贝。详情见 GC3 里的 DSL 解析，见第三节

## 3 GC3 DSL

### 3.1 Buffers and Chunks
GPU 显存在 GC3 里暴露为命名的 buffers，有三种类型：input, 是保存输入数据的地方，output和scratch 这两种未被初始化。GC3 program 的作用是保证每个 GPU 上的 output buffer 都被集合通信的结果准确填充。scratch buffer 被当作临时的存储


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
