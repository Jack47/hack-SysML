细粒度做 checkpointing 的挑战：

1. Checkpointing frequency : 要多频繁地做？
2. Checkpoint stalls : 如何减少 checkpoint 的代价，来让GPU空闲的时间尽量少？ Adaptive rate tuning
3. Data invariant : 如何从 ckpt 正确地恢复过来？

本框架可以提供自动化的，高频的 checkpoint

## 问题
1. 文中提到的 resumable data iterator，现在是什么样子的？
2. 怎么让 ckpt 的代价在几个 iteration 里摊薄？
3. 它如何跟 PyTorch 里的 DALI data loading lib 集成的？
## 启发
1. systematic online profiling
2. data iterator 自动 profiling 几个iter 的耗时，是怎么做的？iter 时间，GPU snapshot 时间，GPU 峰值利用率
3. 恢复的时间：这个怎么换算的？怎么做到从几个小时的恢复时间，缩小到几秒钟的？是不是说明它写入 ckpt 的时间就是几秒钟？

## 参考资料
1. [slides](https://www.usenix.org/system/files/fast21-mohan.pdf)

