## Client

## PatrickStarEngine
这个 engine 是做了一些自动按照配置生成 client，然后调用 modle_func 产出 model，再把这个 module 上挂上钩子的操作，最终生成的是 wrapped module 和 optimizer。当然也可以不用他这个 `init_engine` 函数，

自己做上述事情也可以。

## Warmup 
第一个 iteration 会是用来warmup的，启动 RuntimeMemTracer

## 用法

```
# 配置 patricstar，不填的话会用默认的
config = {
}

model, fp16Adam_optimizer = initialize_engine(
    model_func=model_func, local_rank=get_rank(), config=config
)
```
上述返回的就是一个包裹之后的 model

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
2. 能否不用它提供的优化器？比如用 SGD 
3. memtracer.py 里，1. 其中的 max gpu system mem (non-chunk) 如何统计？总的 GPU 显存使用 - `gpu_chunk_used_mem` . 2. warmup 之后，如何计算当前和下一个 moment 里所需内存？好像就是当前步骤计算出来的。因为model states 跟输入大小无关
4. 什么时候执行 Chunk 的 release？
5. 
