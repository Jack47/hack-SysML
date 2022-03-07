pytorch 1.8 里 syncbn 的实现，forward 和 backward 里都有nccl 通信的过程：

## Forward
1. all gather (mean, invstd, count)
https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/nn/modules/_functions.py#L28

```
dist.all_gather(combined_list, combined, async_op=False) # 这里是阻塞/同步的( work = xxx. work.wait() if not async_op)
```

## Backward

2. 为了计算 input grad 而 all_reduce sum_dy, sum_dy_xmu
https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/nn/modules/_functions.py#L80

##  参考资料： 
[limu 团队：跨卡同步 Batch Normalization](https://zh.mxnet.io/blog/syncbn#batch-normalization%E5%A6%82%E4%BD%95%E5%B7%A5%E4%BD%9C)

[CVPR 2017 tutorial Hekaiming 和 RG 解释了 BN 原理](http://deeplearning.csail.mit.edu/)

[Variants of BN](https://fengweiustc.github.io/paper-reading/2020/06/22/bn/)

[SenseTime 的实现](https://hangzhang.org/PyTorch-Encoding/tutorials/syncbn.html)
