
## 用途
cuda 9.0 之前，CUDA 只支持线程块内的同步：__syncthreads()。这个粒度比较粗，所以 cooperative groups 是后来提出用于更细/大粒度控制同步的

![](./imgs/cooperative_groups.png)
从上图可以看到里面有几个概念，而且左右两侧是不同的级别，而每个级别内部还可以再划分层级

特点：更加灵活、动态分组

用于更方便求解这类问题：

1. 给定一个矩阵，如何快速求出挨个元素求和后的总和？
2. 
## 几个概念

让 groups 成为第一类公民，提高了软件组合性：**collective 函数** 可以 **显示** 得到代表参与的 threads 分组的参数

### thread_block
这个是 cuda 里本来就有的，通过 `cg::this_thread_block()` 来获得，除了 `block.sync()`, 也可以 cg::synchronize(block) 来同步

```
cg::thread_block b = cg::this_thread_block();
cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

```
本文中提到的东西，都在 cooperative_groups 这个 namespace 下

### Collective Operations

Thread groups 上可以进行集合操作：需要同步或者在指定线程集合上通信的操作。每个参与的线程都需要执行这个操作。


可以通过 g.sync() 或者 cg::synchronize(g) 来同步当前的 group

主要好处是当一个kernel 要求传入 `thread_group`，那么里面肯定有同步。


下面是一个例子，在指定的 `thread_group` 里，计算每个 thread 传入的 val 的和



感觉 temp[lane] =val 这个加载没必要放到循环里，加载一次就行？

### thread group

在一个 group 里，可以知道大小：

1. size()
2. thread\_rank: 自己排第几号
3. is\_valid

这些函数都只对置身于自己代表的 group 里的成员线程才有效

下面是 16M 大小数据求和的例子:

```
int n = 1<<24;
int blockSize = 256;
int nBlocks = (n + blockSize - 1) / blockSize;
int sharedBytes = blockSize * sizeof(int); // Block 内部共享的大小

int *sum, *data;
cudaMallocManaged(&sum, sizeof(int));
cudaMallocManaged(&data, n * sizeof(int));
std::fill_n(data, n, 1); // initialize data
cudaMemset(sum, 0, sizeof(int));

sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum, data, n);


using namespace cooperative_groups;
__global__ void sum_kernel_block(int *sum, int *input, int n)
{
    int my_sum = thread_sum(input, n); // 不用 shared memory，也不需要同步的情况下，先计算出4个元素一组的和。其中有3/4的 thread 返回的是0: 不需要它们干活

    extern __shared__ int temp[]; // shared in block
    auto g = this_thread_block();
    int block_sum = reduce_sum(g, temp, my_sum);

    if (g.thread_rank() == 0) atomicAdd(sum, block_sum); // 最终只有 0 号 thread 里是当前 block 里的结果，所以要加到最终的结果里 （nBlocks 个结果要想加）
}
__device__ int reduce_sum(thread_group g, int *temp, int val) // 每次计算出两个元素的和，这样迭代到只省最后一个元素 thread 0 (因为 i 是逐步变小)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) // 每次能对一半的元素数量求和
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane<i) val += temp[lane + i]; // 每次都是加了一半的元素，下标小的 lane 会把大的那半加上
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

__device__ int thread_sum(int *input, int n) // 每4个元素一组，计算出sum
{
    int sum = 0;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n / 4; // 3/4 的 thread 不用干活
        i += blockDim.x * gridDim.x)
    {
        int4 in = ((int4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}
```

优化的空间：

1. 不需要启动 n 个线程，启动 n*1/4即可

### Parittioning Groups / tiled_partition
可以把已有的 group 划分为新的小组，比如 `tiled_partition()`

将一个线程块分解为多个小的协同线程组 (tiled subgroups):

下面的代码执行起来有问题吗？
```
__device__ int sum(int *x, int n) 
{
    ...
    __syncthreads();
    ...
    return total;
}

__global__ void parallel_kernel(float *x, int n)
{
    if (threadIdx.x < blockDim.x / 2)
        sum(x, count);  // error: half of threads in block skip
                        // __syncthreads() => deadlock
}
```

### thread block tiles

## 网格级别的同步

## 多设备同步

## 问题
1. 这个特性页面提到可以支持生产者-消费者并发模式。那以前不支持？ 可能指的是先并发 load，再并发消费的模式，而非一个人不断消费，另一个人生产的模式
