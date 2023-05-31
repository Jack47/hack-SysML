Python backend 好处是直接用 python 来serve models，而不需要写任何的 c++ 代码

实现 execute 时，有两种模式：default mode 和 decoupled mode

### decoupled mode
需要设置 transaction policy 配置。在这个模式下，使用每个请求上有一个的 `InferenceResponseSender` 对象来创建和发送给request的任意数量的响应结果。

Use cases：

InferenceResponseSender 对象可以被传递到单独的先处理，这样主调用方可以从 execute 函数退出，但是模型依然可以给前述的 sender 发送结果

### model config file
config.pbtxt 文件里指定了 backend 的类型，比如 python。
```
models
    add_sub
        1
            model.py
        config.pbtxt # 一个模型配置，覆盖了多个版本的模型
```

### Creating Custom Execution Environments
如果每个python model 想用自己的执行的python环境，那可以使用自定义的。当前只支持 conda-pack。它会保证环境是可移植的。

config.pbtxt
```
name: "model_a"
backend: "python"

...

parameters: {
    key: "EXECUTION_ENV_PATH",
    value: {string_value: "/home/iman/miniconda3/envs/python-3-6/python3.6.tar.gz"}
}
```

### Multiple Model Instance Support
Python 解释器里使用了全局的锁叫 GIL，所以多线程没法在 python 解释器里并行跑，因为每个线程都得拿到 GIL。为了绕过这个限制，Python backend 可以 spawns 一个单独的进程给每个模型实例(model instance)。这个和 ONNXRuntime，TF，PT 等后端处理多模型实例是不同的。他们是多线程

### Business Logic Scripting
业务逻辑和模型执行组合的形式叫 BLS

## TODO：
1. 看下 comoplete example for sync and async BLS in python backend: examples
## 问题
1. 我们现在的 tritonserver 启动时，`--model-repository ${PWD}/models` 这个值是什么？
2. TritonPythonModule 里的 initialize 里，传入参数长什么样子？
3. 看看 modelmesh 里的例子，使用的 python backend 还是什么？
