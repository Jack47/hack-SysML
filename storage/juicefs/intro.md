![](imgs/read-cache.png)
1. 有三次内核态和用户态之间的转换
2. 两个Page Cache 都在内存里，但是第一个是文件的 page cache，第二个是 Block 级别的 page cache
3. 从图里看到几个级别的 cache，都在本地。remote 里只有 metadata engine 和 Object Storage
