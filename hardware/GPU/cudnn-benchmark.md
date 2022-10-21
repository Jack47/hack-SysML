## cudnn 接口
cudnnFindConvolutionBackwardFilterAlgorithmEx()

如何判断哪个算法最快？

主要是对卷积有很好的效果？

### cudnn 里打印选取的最好的几个结果的地方
https://gitlab.bj.sensetime.com/linklink/pytorch/blob/v1.11.0/caffe2/operators/op_utils_cudnn.h#L35-44

默认认为返回结果里第0个就是最好的

## try_all
调用 try_all 的就是上层的conv逻辑，里面会在fwd or bwd时调用

一旦发现这个算法的参数之前测过在cache(hashmap)里，就直接用[之前的结果](https://gitlab.bj.sensetime.com/linklink/pytorch/blob/v1.11.0/aten/src/ATen/native/cudnn/Conv_v7.cpp#L504-505)：

否则去找最好的，用的cudnn的接口

本身上述代码并没有跟具体的算法强绑定。

其中 conv 上的cache 时判断的参数是这些 [ConvolutionParams](
https://github.com/pytorch/pytorch/blob/24fd84313f010822ceb62166b46b9ca61d902fc9/aten/src/ATen/native/cudnn/ConvShared.h#L18-L32)
## search::findAlgorithms

## 参考资料：

[What does torch backends cudnn benchmark do](https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)
