## 一些名词解释：

Windows On Snapdragon: SC8280X( 高通的芯片代号），这个是个 Windows 系统的平板。有个缩写 WoS(Windows on Snapdragon)

HTP: Hexagon processor with fused AI accelerator architecture



分为三个步骤：

## 1. onnx 模型的优化，使用 qnn-onnx-converter
qnn 就是 qualcomm neural network(QNN) 是 Qualcomm AI Engine Direct SDK 的同义词

qnn-onnx-converter 工具可以把模型从 ONNX 转换为对应的 QNN 格式，而且是 A16W8 的精度。这个步骤会产生 .cpp 文件，代表了一个由一系列 QNN API 调用组成的模型，以及一个 .bin 文件，包含了静态数据，通常是模型权重，被 .cpp 文件引用


## 2. QNN Model Library

qnn-model-lib-generator 会把 .cpp 和 bin 文件编译到一个给定目标上的共享对象库的文件里。输入就是上一步输出里的两个文件


## 3. QNN HTP context binary

qnn-context-binary-generator 可以用来把 libqnn_model.so 转换为 qnnhtp.serialized.bin，是一个可以被 QNN HTP 后端所执行的二进制程序
这一步骤需要上一步的输出，也需要 libQnnHtp.so 文件在 Qualcomm AI Engine Direct SDK

最终用 qnn-net-run 工具来运行上述的 bin 文件

本文主要来自这里：https://docs.qualcomm.com/bundle/publicresource/topics/80-64748-1/introduction.html
