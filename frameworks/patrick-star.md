## 用法

```
# 配置 patricstar，不填的话会用默认的
config = {
}

model, optim = initialize_engine(
    model_func=model_func, local_rank=get_rank(), config=config
)
```
## 从 issue 和文档看到的一些信息

我们真的需要模型并行(MP)么？https://github.com/Tencent/PatrickStar/issues/13#issue-972345430

CV 里，需要用 Adamw。这个已经是支持的
https://github.com/Tencent/PatrickStar/issues/228#issuecomment-969973622

原来是通过 module 的 hook 来采样 cuda/cpu 的内存，这个不够准确，后来是起了一个线程，0.01 采样一次
https://github.com/Tencent/PatrickStar/issues/190

seq2seq, LSTM 模型里会有参数被更新两遍的情况，但它们没有太大的模型

profile 的结果：
https://github.com/Tencent/PatrickStar/issues/33#issuecomment-940730316

讨论 PatrickStar 是否要做成 DS 的插件
https://github.com/Tencent/PatrickStar/issues/12

Embedding 是内存密集性的算子

Hybrid ADAM: 对一些小模型有用：一些计算在 CPU 上做，同时剩下的计算在 GPU 上

--with_async_mem_monitor: 异步，关掉就是同步的，类似jianjin 用的那个东东


with_static_partition： 可以在 CPU 和 GPU 之间静态划分模型数据

修改 CPU 和 GPU 内存跟踪器

overall_gpu_mem_ratio: 可以调大一些，如果遇到 OOM
overall_cpu_mem_ratio:

margin_use_ratio

## 问题
1. 如果 CV 里用，需要是 Adam，如果不是，需要自己实现优化器，及其上的更新
