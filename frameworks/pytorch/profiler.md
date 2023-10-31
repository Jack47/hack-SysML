## 1 [PyTorch 2.1 里的 Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): torch.profiler
可以分析 CPU、GPU 执行时间，也可以分析显存占用，有 trace 功能，有 stack trace

```
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof: # 加这两行就行
    with record_funciton("model_inference"):
        model(inputs)
```

结果都是 pytorch 里算子级别的，比如 aten::conv2d 这种

## 2 autograd.profiler
```
with torch.autograd.profilerr.profiler() as prof
```
