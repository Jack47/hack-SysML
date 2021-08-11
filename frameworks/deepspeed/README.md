## TODO
1. contiguous_memory_allocator.py
2. weight_quantizer.py
3. stage1.py
4. partition_parameters.py
5. nccl.py : compressed_allreduce 实现

## 看源码想搞清楚的问题
1. DeepSpeed Profiler performance tool 是怎么实现的: 如果做到诊断出 gaps？是简单跟峰值做对比？如何识别出性能瓶颈？用火焰图？细粒度到 sublayer，单个操作。这样可以看是否用 fused kernel 来减少单独调用 kernel 的损耗，或人工优化
2. 这个框架里有没有什么可以借鉴的地方
3. Effective quantize-aware training: 有哪些是 DS 实现，哪些是 NV 提供的。
4. Compressed training : coarse-grained sparsity in Transformer layers via Progressive Layer Dropping : 在每个迭代里，基于**进步的调度(progressive schedule)**，考虑模型的敏感性，沿着临时和深度两个维度来做动态关闭(跳过)某些 transformer layer 的操作(forward-backward)。这个方向专门有论文：[Progressive Layer Dropping](https://www.microsoft.com/en-us/research/publication/accelerating-training-of-transformer-based-language-models-with-progressive-layer-dropping/)

## Flops Profiler
不需要用户做改动，就可以在 DS runtime 里开启，也可以是单独的包。支持用户自定义的 module，原因是可以捕获 torch.nn.functional 的调用来预估出 flops 和 params 的数量

from deepspeed.profiling.flops\_profiler import get_model_profile


## 启发

1. 我们可以用这个轮子，自动算出 flops，以及每一层的 flops
2. [获取 hostname 的方法](https://github.com/microsoft/DeepSpeed/blob/504893aea40004cf9916ddc3ca0ddbfd0e784c8d/deepspeed/utils/distributed.py?plain=1#L66)
3. [获取并打印 torch, nvcc 等版本的方法](https://github.com/microsoft/DeepSpeed/blob/504893aea40004cf9916ddc3ca0ddbfd0e784c8d/deepspeed/env_report.py?plain=1#L84)
