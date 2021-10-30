
reduce 的作用是把一个数组里的值最终搞成一个，比如要求 sum，mean，variance 等，使用的是双目运算符。

cuda 里已经提供的一些 reduce 能力：

## Warp Shuffle Functions
exchange a variable between threads within a warp. 它好处是不使用共享内存

__shfl_xor_sync

其中有几个概念：lane, 
