
## 1. 快速开始的检查列表
下面是具体的一些针对卷积层的快速检查列表

* 选择让 input 和 output 的 channels 被8（fp16）或者4（TF32）整除，目的是让 Tensor Core 发挥作用。大部分 CNN 里第一个卷积层由3个channel的卷积层组成，如果 stride 为2，把3 channel 填充到 4 更高效，具体见 输入输出通道一节
* 选择至少被64，最理想是256整除的参数(batch size, number of input and output channels)会让 **tiling更高效***，reduce的开销更小。见 量化效果一节
* 尺寸相关的参数(batch size, input and output height and width, number of input and output channels)，更大的值能提高并行度。对于全连接层，这个会加速算子的效率，但不会减少绝对的时长。见卷积参数如何影响性能
* NV的库提供了一套不同的卷积算法，各自的性能表现不同，取决于卷积的参数。当网络的输入大小每个迭代都相同，可以启用 autotuning 来选择理想的算法。对于 TensorFlow，autotuning 默认开启。对于 PyTorch，可以使用 torch.backends.cudnn.benchmark = True. 这个只是适合每次迭代之间数据维度不变的情况：第一次会用不同 cuDNN 实现，之后会缓存每个的速度情况，后续选择最快那个。但如果输入维度经常变，那每次遇到一个新的，都会进行benchmark，[就会拖慢速度](https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)
* 选择内存中 tensor 布局来避免转置输入和输出数据。有两类约定: NHWC 和 NCHW。推荐尽可能用 NHWC 格式。见 Tensor Layouts in Memory: NCHW vs NHWC

Tensor Core里，NHWC(即挨个像素进行存储)这种情况速度会更快，如果不转换，conv 算子内部会自动转换。pytorch里可以使用如下方式指定：`x = ytensor.to(memory_format=torch.channels_last); model.to(memory_format=torch.channels_last)`。既要让数据修改，也得让模型也知道这个变化，所以是两处修改。这种格式转换在 fp16 ，启用 tensor core 的情况下性能受益最大：resnet50 1.22 倍。但目前并不是[所有算子都支持 Channels Last](https://github.com/pytorch/pytorch/wiki/Operators-with-Channels-Last-support)，有文档可以参考者[自己实现](https://github.com/pytorch/pytorch/wiki/Writing-memory-format-aware-operators)。


## 输入输出通道(Channels In And Out)
在 cuDNN v7.6.3(V100) 和之后，卷积维度会自动被对齐，来方便利用到 Tensor Core。更早期版本比较严格：在 NHWC 数据上使用 TC，需要 C 和 K 都对齐到 8(fp16) 或者 4(tf32)

## 量化效果(Quantization Effects)


Tiling 是说输出矩阵被划分为给定的大小的块，然后这些块被分发到 multi-processor 上进行计算。比如 A100 GPU里如果有108个 SMs，那么每个SM并发处理一个 thread block；为了最大化并行， GEMM 操作里应该包含 108的倍数个小块。否则最后需要单独拿出几个 SM 来初一一波最后的无法被108整除的那几个小块数据。

## 卷积参数如何影响性能


上述主要来自，此系列的[索引页: NVIDIA Deep Learning Performance](https://docs.nvidia.com/deeplearning/performance/index.html)
