优势是类似 tfrecord，可以把随机读取转换为顺序读取，对磁盘和文件系统友好

这里提到的数据集 shuffle 的问题，webdataset 作者也给出了解法：https://github.com/webdataset/webdataset/issues/71

TODO:

官网上有好几个 youtube 视频可以看看

它之所以快，是因为对 OS 而言：
1. 减少了 I/O 操作的次数: 怎么做到的？也是因为不随机读取了？
2. 可以提前读取数据(顺序读)

## fio (Flexible I/O Tester)
测试存储设备性能的工具，可以模拟各种 I/O 负载

1. 基本随机读测试
```
fio --name=randread --ioengine=libaio --iodepth=16 --rw=randread --bs=4k --size=1G --numjobs=1 --runtime=30s --time_based
 --direct=1
```
--iodepth=16: I/O 深度为16，即同时进行的 I/O 操作数量
--numjobs=1: 并发作业数为 1
--runtime=30s
--time_based: 测试是基于时间的，而不是基于完成的 I/O 操作数
--direct=1: 启用直接 I/O，绕过文件系统缓存

2. 随机写入测试
```
fio --name=randwrite --rw=randwrite --randrepeat=1
```

--randrepeat=1 : 是否设置随机模型；1是完全随机，0是顺序

3. 混合随机读写测试
--rwmixread=70: 设置读写混合比例，70% 为读取，30% 为写入

4. 使用文件作为测试源
--filename=/path/to/testfile

