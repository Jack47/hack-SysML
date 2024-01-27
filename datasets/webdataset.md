优势是类似 tfrecord，可以把随机读取转换为顺序读取，对磁盘和文件系统友好

这里提到的数据集 shuffle 的问题，webdataset 作者也给出了解法：https://github.com/webdataset/webdataset/issues/71
