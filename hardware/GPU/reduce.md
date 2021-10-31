
reduce 的作用是把一个数组里的值最终搞成一个，比如要求 sum，mean，variance 等，使用的是双目运算符。

cuda 里已经提供的一些 reduce 能力：

## Warp Shuffle Functions
exchange a variable between threads within a warp. 它好处是 **不使用共享内存**。可以用来：

1. 把单个线程里的数据广播给其他人
2. 可以按照各自的分区算加法
3. 在 warp 里做规约

### 1. Broadcast of a single value across a warp
value = __shfl_sync(0xffffffff, value, 0); // synchronize all threads in warp, and get "value" from lane 0. 这句会在除了 lance 0 之外的线程里执行


__shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize) # 会用到 butterfly reduction，好处是可以跨 warp 来规约

mask 的作用:

其中有个概念：lane. lanes: 一个 warp 里的线程，被叫做 lanes，会有一个[0-warpSize-1] 的下标

## 疑问
1. 这个实现是高效的吗？为何
2. lane 不是物理上的概念吗，sm, core, lane？
3. butterfly reduction 是啥？

