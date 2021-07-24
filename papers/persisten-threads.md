
## 其他参考资料：
[Persistent RNNs](https://hgpu.org/?p=16050) : 通过 Persistent Threads 来实现更高的计算吞吐，原因是减少了16倍的激活值内存大小：通过cache 这些权重，在多次推理过程中复用
