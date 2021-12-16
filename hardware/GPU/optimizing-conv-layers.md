
## 1. 快速开始的检查列表
下面是具体的一些针对卷积层的快速检查列表

* 选择让 input 和 output 的 channels 被8（fp16）或者4（TF32）整除，目的是让 Tensor Core 发挥作用。大部分 CNN 里第一个卷积层由3个channel的卷积层组成，如果 stride 为2，把3 channel 填充到 4 更高效，具体见 输入输出通道一节
* 选择至少被64，最理想是256整除的参数(batch size, number of input and output channels)会让tiling更高效，reduce的开销更小。见 量化效果一节
* 尺寸相关的参数(batch size, input and output height and width, number of input and output channels)，更大的值能提高并行度。对于全连接层，这个会加速算子的效率，但不会减少绝对的时长。见卷积参数如何影响性能
* NV的库提供了一套不同的卷积算法，各自的性能表现不同，取决于卷积的参数。当网络的输入大小每个迭代都相同，可以启用 autotuning 来选择理想的算法。对于 TensorFlow，autotuning 默认开启。对于 PyTorch，可以使用 torch.backends.cudnn.benchmark = True
* 选择内存中 tensor 布局来避免转置输入和输出数据。有两类约定: NHWC 和 NCHW。推荐尽可能用 NHWC 格式。见 Tensor Layouts in Memory: NCHW vs NHWC


## 输入输出通道(Channels In And Out)

## 量化效果(Quantization Effects)

## 卷积参数如何影响性能
