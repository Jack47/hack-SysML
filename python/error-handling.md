之所以使用异常而非返回值的原因：这样代码更容易读和维护，因为错误处理都在那些 exception 部分，正常逻辑流程里可以不用关心，否则就需要每个函数调用之后立马 if else

```
try:
    handle = getHandle(EV1)
    record = retrieveDeviceRecord(handle)
except DeviceOpenEror as e:
    pass
except DeviceRetrieveError as e:
    pass
except DeviceCloseEror as e:
    pass

```
 try 代码块的秒处：在程序中定义了一个范围：try 部分像一个事务，catch 将程序维护在一种状态

## 最佳实践
0. 抛出异常时带上环境上下文的信息，比如失败的操作和类型，传递足够的信息，调用者会记录下来。抛出异常时就想象调用者如何处理它！

1. 建议是捕获具体的异常类型，这样能知道如何处理这些异常

bad: 使用通用的一个异常

good：针对不同的异常类型做处理：

```
except HTTPSenderException:

except WatermarkException:

except InferencePipelineException:

```

2. 如果不知道如何处理当前发生的异常，最好是传播到上层调用栈里

3. 使用需要释放的资源比如文件、网络句柄时，使用 context manager (with 语句）
```
with open("file.txt") as f:
    # Do some thing with the f

# File is automatically closed here, even if an exception was raised
```

4. 别返回 null 值
而是返回一个 EmptyList，这样调用者就不用额外单独判断是否为 null，否则会增加工作量

5. 创建一个类来处理特例（比如为空）

## 定制自己的异常
当 Python 里自带的不够具体，就自定义：
```
class BatchSizeNotExpectedException(Exception):
    def __init__(self, bs):
        self.bs = bs
    
    def __str__(self):
        return f"batch size expected 1, but got {self.bs}"
    
raise BatchSizeNotExpectedException(bs)
```


## 看下后端对返回结果的处理

## 看下推理服务对异常的处理


