之前是只用于推理，现在开始逐步支持训练了： [BERT training on IMDB movie sentiment](https://github.com/microsoft/nnfusion/pull/199): 21年1.20号
不过有很多限制：fixed lr sgd optimize, dropout disabled(kernel not ready)

目前只支持 TensorFlow 和 ONNX 格式的模型输入
