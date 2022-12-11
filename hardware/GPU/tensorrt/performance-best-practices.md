摘自官方文档：https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#performance

## 13 Performance Best Practices

### 13.1 Measuring Performance
#### 13.1.2 CUDA Events
优先推荐用 [cudaEvent(例子）](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#cuda-events) 来测量，不然直接用 host 和 device 的synchronize 粒度太粗，而且如果是并行推理，会影响其他GPU上的计算kernel

#### 13.1.3 Built-In TensorRT Profiling
有C++,Python接口：IExecutionContext::setProfiler()，或者使用 trtexec 

#### 13.1.4 CUDA Profiling Tools
C++、Python里可以api控制 trt.ProfilingVerbosity.NONE/DETAIL

```
trtexec --profilingVerbosity=detailed --onnx=foo.onnx --saveEngine=foo.plan # 在生成 engine 时控制
```
### 13.2 Hardware/Software Environment for Performance Measurements
GPU Clock Locking and Floating Clock

GPU Power Consumption and Power Throttling

H2D/D2H Data Transfers and PCIe Bandwidth

### 13.3 Optimizing TensorRT Performance
#### 13.3.1 Batching
更大的 batchsize，经常带来更高的 GPU 处理。如果网络里有 Matmul 或者全连接层，32的倍数的 batch size 会让 fp16 或者 int8 的推理更高效，因为使用上了 TensorCores

有些时候由于程序组织的原因，没法做batch的推理。此时如果是一个server 在对每个请求做一次推理，那此时可以实现看机会的batching。每个来的请求，等待T时间，如果此件有其他请求来，就
批量处理他们。否则，依然用一次推理。这种方式给每个请求增加了一定延迟，但是可以数量级提升系统的最大吞吐量

#### 13.3.2 Streaming
在TensorRT的推理场景下，资源利用率不高的情况下，可以用多strema来跑：
1. 创建网络的单个的引擎
2. 给每个batch的请求创建单独的 cudastream
3. 异步发送inference的请求
4. 当所有工作发射后，等待所有的stream完成
使用多个并行的stream后，他们之间共享资源，这样TRT可能在选择某个kernel时，会使用当前实际运行情况下的次优的kernel实现。为了减轻这种现象，可以限制资源使用

通常是用一个worker的线程池来服务进来的请求，这种情况下，每个线程有一个执行contxt和cuda stream。每个线程请求在自己的 stream 里干活。这样它自己synchronize 自己的stream，
不影响其他线程

#### 13.3.3 CUDA Graphs
#### 13.3.4 Enabling Fusion
具体详情可以看 detail 日志？并没有看到有什么开关来控制

#### 13.3.5 Limiting Compute Resources
比如限制为只用最大50%的算力

Using Timing Cache: 可以记录每个layer上配置的信息和对应的延迟，这个可以让后续 build engine 时复用之前的速度结果，速度能更快

#### 13.3.6 Deterministic Tactic Selection
Engine Inspector 或者 build engine 时的详细日志能有帮助

#### 13.3.7 Transformers Performance
有几种方法来运行 Transformer 网络，包括 native TensorRT(using ONNX)，使用 TensorRT OSS demoBERT 插件的例子，或者使用 FasterTransformer。各自有不同的收益和用户场景

Native TensorRT 可以在不改变代码的情况下，提供最大的灵活度来fine-tune 网络的结构和参数，而 demoBERT 和 FasterTransformer 聚焦在特定的网络上，而且需要手工更新配置甚至是网络上改变。使用 Native TensorRT后，可以使用 Triton Inference Server 来**无缝部署**推理服务

Supports INT8:  TensorRT 的 ONNX Parser：Yes， demoBERT Yes，FasterTransformer：yes


#### 13.3.8 Overhead of Shape Change and Optimization Profile Switching

### 13.4 Optimizing Layer Performance
### 13.5 Optimizing for Tensor Cores
### 13.6 Optimizing Plug-ins
### 13.7 Optimizing Python Performance
### 13.8 Improving Model Accuracy
### 13.9 Optimizing Builder Performance
TRT支持使用 FP32、FP16和INT8精度来执行一个层（取决于builder配置）。默认使用更优性能的精度。这样有些情况下导致精度差。通常使用更高精度可以帮助提高精度，而性能稍差

有三个步骤来提高模型精度：

#### 1. Validate layer outputs: 
a. 使用 Polygraphy 来 dump layer 输出并验证没有 NaN 或者 Infs。`--validate` 选项可以检查 NaN 和 Infs。因此，我们可以比较layer的输出
b. 对于 FP16，模型可能需要 retraining 来确保中间层的输出可以被表示为fp16的精度，而不需要 overflow、underflow
c. 对于 INT8，考虑重新在更丰富的校准数据集上进行重新校准. 对于PyTorch 上训练的模型，除了 TRT 里的训练后量化(PTQ)，还有 NV的PyTorch上量化套件来做量化感知训练(QAT)。可以从两种方法里选择更快的那一种

#### 2. 操作层的精度
a. 一些情况下，某些层在特定精度上导致结果出错，可能是因为这个层固有(inherent)的限制（比如 LayerNorm 输出不能是 INT8），模型限制(output gets diverged resulting in poor accuracy), 或者给 TRT 一个bug
b. 可以控制每个layer的**执行精度**和**输出精度**
c. 实验性质的 [debug precision](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy/tools/debug) 工具可以帮助找到需要用更高精度运行的layer

## 参考资料：
1. [PyTorch下使用 ONNX 、 TRT 推理的例子](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb)
