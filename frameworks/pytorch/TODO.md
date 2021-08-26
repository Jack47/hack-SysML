## 疑问
这个是干嘛的？[RFC-0012: Functional lazy traces from XLA to PyTorch](https://github.com/pytorch/rfcs/pull/18)。这个 trace 跟 torchscript 里的 trace 是同一个？ 意味着 pytorch 里也支持 xla 了么？ 为什么在 dtr 里提到了呢？

好像是为了让 pytorch 能跑在 TPU上

## TODO
1. 看看 pytorch  extension 机制，是不是 amp 就这么做的


