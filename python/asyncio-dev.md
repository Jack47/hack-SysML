本文参考自这里：https://docs.python.org/3/library/asyncio-dev.html

异步变成和传统的“串行”非常不同。本页主要列出了常见错误和陷阱，解释了如何避免他们

## Debug Mode
默认情况下 asyncio 运行在生产模式。为了轻松开发，asyncio 有一个 debug 模式。 

有几种方法来开启 debug asyncio 模式:

* PYTHONASYNCIODEBUG=1
* 使用 Python Development Mode: 3.7 开始，引入一个运行时检查，这个检查代价比较高，所以默认没有开启。
* asyncio.run(debug=True)
* loop.set_debug()

除此之外，还可以：

* 设置 asyncio logger 到 logging.DEBUG
* 设置 warnings 模块来展示 ResourceWarning 的 warnings

debug 模式的好处：

* asyncio 会检查没有进行 awaited 的 coroutines 并记录日志。这个缓解因为“忘记 await”导致的问题
* 很多非线程安全的 asyncio APIs（比如 loop\.call_soon() 和 loop\.call_at() 方法），如果被错误的线程调用了，会抛出异常
* 如果处理 I/O 操作超时，会记录 I/O selector 的执行时间
* 超过100ms 的 Callbacks 会被记录到日志里。The loop.slow\_callback\_duration 属性可以设置最小的执行时间

## 并发和多线程
事件循环(event loop)运行在一个线程里（通常是主线程），在它的线程里执行所有的回调(callbacks)和任务(tasks)。当任务在事件循环里执行时，其他任务无法在这个相同的线程里执行。当一个Task执行 await 的表达式，运行的任务会被挂起，事件循环执行下一个任务

当调度一个来自其他 OS 线程里的 callback 时，必须使用 loop.call\_soon\_threadsafe() 方法。例子：
```
loop.call_soon_threadsafe(callback, *args)
```

几乎所有的 asyncio 的对象都是非线程安全的，通常不是问题，除非有代码是在一个 Task 或者回调外面去使用他们。如果真的需要这样做，那就调用底层的 asyncio API。

调度一个来自其他 OS 线程里的 coroutine 对象时，需要使用 run\_coroutine\_threadsafe()

```
async def coro_func():
    return await asyncio.sleep(1, 42)
# Later in another OS thread:
future = asyncio.run_coroutine_threadsafe(coro_func(), loop)

# Wait for the result:
result = future.result()
```

为了处理信号和执行子进程，event loop 必须运行在主线程里。

loop.run\_in\_executor() 方法可以搭配 concurrent.futures.ThreadPoolExecutor 来在不同的 OS thread 里执行阻塞的代码，避免阻塞 event loop 所在的线程

## 执行阻塞的代码
阻塞（CPU-bound而非iobound）的代码不应该直接调用它。比如，如果一个函数执行了 CPU 密集的运算一分钟，那么所有并发的 asyncio 任务和IO操作都会被延迟1秒

可以使用 executor 来在不同的线程甚至不同的进程里执行，来避免把 event loop 的主线程阻塞住。参考 loop.run\_in\_executor()

## logging
```
logging.getLogger("asyncio").setLevel(logging.WARNING)
```

## 检测从来不会被 await 的coroutines

一个 coroutine 函数被调用，但是没有被 awaited （比如调用 `coro()` 而非 `await coro()`)，或者 coroutine 没有被用 asyncio.create_task() 调度，那么 asyncio 会 emit 一个 RuntimeWarning

所以 async def 的意思是下面的函数是 coroutine，调用时会被当做 task？
