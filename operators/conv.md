一个卷积层的输出形状是由输入、kernel 大小，zero padding和strides来共同决定的。而跟 fc 对比，fc 里输出跟输入是没关系的。

一个离散卷积，有以下特性：

1. 离散（只有几个输入单元对输出单元有贡献: 是因为kernel里可能有0的情况？）
2. 复用参数：kernel 里同样的权重会用在多个输入的位置上(共享同样的参数 )

## 参考资料
1. [A guide to convolution arithmetric for deep learning]
