## 为什么需要 profiling？
能帮助分析出来瓶颈，比如一个函数执行了多久，用了多少内存

## simple : 发现训练大循环里的瓶颈
比如是读取数据，还是计算耗时大

```
trainer = Trainer(profiler='simple')
```
## profile 每个函数里的时间
需要用 AdvancedProfiler，它基于 Python 的 cProfiler：
```
trainer = Trainer(profiler='advanced')
```
一但 `.fit()` 执行完，就可以看到这样的输出：
```
Profiler Report

Profile stats for: get_train_batch
        4869394 function calls (4863767 primitive calls) in 18.893 seconds
Ordered by: cumulative time
List reduced from 76 to 10 due to restriction <10>
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
3752/1876    0.011    0.000   18.887    0.010 {built-in method builtins.next}
    1876     0.008    0.000   18.877    0.010 dataloader.py:344(__next__)
    1876     0.074    0.000   18.869    0.010 dataloader.py:383(_next_data)
    1875     0.012    0.000   18.721    0.010 fetch.py:42(fetch)
    1875     0.084    0.000   18.290    0.010 fetch.py:44(<listcomp>)
    60000    1.759    0.000   18.206    0.000 mnist.py:80(__getitem__)
    60000    0.267    0.000   13.022    0.000 transforms.py:68(__call__)
    60000    0.182    0.000    7.020    0.000 transforms.py:93(__call__)
    60000    1.651    0.000    6.839    0.000 functional.py:42(to_tensor)
    60000    0.260    0.000    5.734    0.000 transforms.py:167(__call__)

```
如果文件太大，可以stream到文件里：
```
from lightning.pytorch.profilers import AdvancedProfiler

profiler = AdvancedProfiler(dirpath='.', filename='perf_logs')
trainer = Trainer(profiler=profiler)
```
## 构建自己的 profiler
from lightning.pytorch.profilers import Profiler

```
class ActionCountProfiler(Profiler):
    def __init__(self, dirpath=None):
    
    def start(self, action_name):

    def stop(self, action_name):
    
    def summary(self):
    
    def teardown(self):
        ...

trainer = Trainer(profiler=ActionCountProfiler()) # 传入一个实例
trainer.fit(...)

```
## Profile 自定义的代码
需要再 LightningModule 里引用一个 profiler：
```
from lightning.pytorch.profilers import SimpleProfiler, PassThroughProfiler

class MyModel(LightningModule):
    def __init__(self, profiler=None):
        self.profiler = profiler or PassThroughprofiler()

    def custom_processing_step(self, data):
        with self.profiler.profile("my_custom_action"): # 用来标识一段开始
            ...
        return data
        
        
        
profiler = SimpleProfiler()
model = MyModel(profiler) # 为什么 model 和 trainer 里都需要塞进去
trainer = Trainer(profiler=profiler, max_epochs=1)
```

如果是为了 profile 你的任意代码，需要使用 self.profiler.profile()。如上

## Profile pytorch operations
为了理解每个 PyTorch 算子的代价，需要使用 [`PyTorchProfiler`](https://pytorch.org/docs/master/profiler.html)
```
from lightning.pytorch.profilers import PyTorchProfiler


```
##
## 问题
lighting profiler 和 PyTorchProfile 有何区别和联系？


