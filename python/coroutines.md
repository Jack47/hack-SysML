## coroutine
它是更泛化的 subroutines。subroutine 可以在某个时刻进入，在某个时刻退出。Coroutines 可以在不同的时间点被进入、退出、恢复。
Coroutines 可以被 async/await syntax 定义, is the preferred way of writing asyncio applications.

```
async def func(param1, param2): # async def 定义了这个函数为 coroutine functionsss
    do_stuff()
    await some_coroutine()
```
### callback
A subroutine 函数，可以当做参数传递，会在未来某个时刻执行


## 参考资料
1. [Coroutines and Tasks](https://docs.python.org/3/library/asyncio-task.html#timeouts)
