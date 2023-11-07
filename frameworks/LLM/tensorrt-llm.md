# Architecture
是一个工具包，组装了优化后的解决方案来执行 LLM 的推理。它的能力：
1. 提供了 Python API 来定义模型、转换模型到在 NV GPU 上能高效执行的 **TensorRT 引擎**
2. 包含 Python、C++ 的组件来构建执行这些引擎的**运行时**(提供 beam-search，大量的采样函数比如 top-K)，同时借助 Triton Inference Server 可以很容易创建一个基于 web 的 LLM 服务
3. 支持多 GPU 和多节点配置（通过 MPI）

上面的功能可以划分为以下几个步骤：

## 1 模型定义(使用tensorrt 里的定义）

## 2 编译（为 tensorrt engine)

### 权重绑定（加载）

### Pattern-Matching and Fusion
1. 识别出可以 fusion 的系列算子

### Plugins
即使 TRT 有强大的模式识别算法，支持很多 fusion 类型，但是依然有风险是识别不出来非常规或者更高级的模式。所以支持用户自定义实现 plugins

## 3 Runtime
Runtime 组件会把 TensorRT 引擎加载进来，然后驱动执行。通常对于 GPT 这样的模型，运行时负责
加载 engine，然后对输入序列进行处理，还有生成的loop里的body处理
### 多 GPU 和多节点支持
虽然 TRT 只是给单 GPU 系统使用的，但是 TRT-LLM 增加了对多 GPU 和节点的支持。容许使用 TRT 插件来把 NCCL 库的通信原语包在插件里。

这些通信原语有 python 的版本

多 GPU 支持可以通过两种模型并行来开启：TP 和 PP

## In-flight Batching
TensorRT  LLM 支持 in-flight batching 请求（也叫做 continuous batching 或者 iteration-level batching）来提供更高的吞吐。可以参考 Batch Manager 文档

## 问题：
是整个模型一个 engine 还是分开的？
