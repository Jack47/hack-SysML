## Batch Norm
BN 里满足以下条件，就可以开启 semi-persistent NHWC kernel：

* 所有tensor，都是 NHWC packed，而且数据类型是 `CUDNN_DATA_HALF`
* 输入参数 mode 必须是 `CUDNN_BATCHNORM_SPATIAL_PERSISTENT` # cudnn 8.0 里默认就会用
* HWC 格式

[出处](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormalizationForwardTrainingEx)

## Tensor Core DL Performance Guide

对于使用除了 *\_ALGO_WINOGRAD_NONFUSED 算法之外，当下列条件满足，cuDNN lib 就出发 Tensor Core 操作：

* 输入、filter和输出都是 `CUDNN_DATA_HALF`(包含bfloat16？)
* 输入和输出 feature map(比如 channel 维度)是8的倍数。当通道维度不是8的倍数，可以用 padding
* filter 类型是 `CUDNN_TENSOR_NCHW` OR `CUDNN_TENSOR_NHWC`
* 如果 filter 用的 `CUDNN_TENSOR_NHWC` ，当输入，filter和输出数据都被对齐到 128 bit(16B) 边界

[CUDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#tensor-ops-conv-functions-data-filter-formats)

[Deep Learning Performance Optimization with **Profiling Tools**](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31228/): 主要介绍 DLProf, 里面有 Tensor Core Efficiency ，可以看出有多少可用 TC，但是没用到
