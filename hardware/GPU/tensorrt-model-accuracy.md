来自这里：https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#model-accuracy

TensorRT 可以根据 builder 配置来让某个层以  FP32、fp16 或者 int8 精度来执行。默认是取最佳性能下所需的精度。有时会导致精度很差。通常情况下，可以用更高精度来执行某个layer，牺牲一些性能。

有几个步骤可以提高模型精度：

## 1. 验证层的输出
a. 使用 **Polygraphy**(一个小工具，可以在多种框架里运行，然后debugging DL 模型。可以比较推理时每层结果） 来dump 层的输出，验证是否有 NaNs 或者是 Infs 的值。`--validate` 选项能检查这两种异常。同时，也可以和一些标准的值比如ONNX runtime里的这个layer的输出值做对比
b. 对于 FP16，可能模型需要重新训练来确保中间层的输出都可以在fp16精度下无上溢出、下溢出地表示。
c. 对于 int8，考虑使用更具有代表性的校准数据集来做校准。pytorch里已经有一个 QAT 下的 PTQ TRT
## 2. 操作层的精度
a. 有时某个层运行特定精度会导致结果错误。可能是由于没有考虑到 layer里限制（比如 LN 输出不应该是int8），模型限制（输出在低精度下质量很差），或者提个 TRT bug
b. 可以精准控制layer级别的**执行精度**和**输出精度**
c. 有个实验性质的 [debug precision 工具](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy/tools/debug)，可以帮助自动发现需要在高精度下运行的层
## 3. 使用[算法选择和复现套件](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#algorithm-select)来关闭 flaky tactics
a. 当精度在不同的build+run下不一样，可能就是因为上述原因，导致某些层的策略不对
b. 使用算法选择器来dump出精度好和精度坏两种情况下的策略有什么不同。可以配置策略选择器只选择给定的一些子集（比如只容许精度好的那次里的策略）
c. 可以使用 Polygraphy 来[自动化](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/debug/01_debugging_flaky_trt_tactics)这个过程

多次运行之间的精度，不应该改变；一单engine已经给特定GPU上编译好了，就不应该导致多次运行之间有精度变化
