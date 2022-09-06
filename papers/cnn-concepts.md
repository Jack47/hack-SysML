## Feature (activation of a neuron, feature detector)
当一种特定的模式(pattern or feature)呈现在输入里，此神经元就被激活。

神经元识别的模式可以被这样可视化：

1. 优化输入区域来最大化神经元的激活（deep dream）
2. 可视化在神经元激活值的输入下的梯度或者guided梯度
3. 在训练集合上可视化一堆图片的区域，这些区域激活了这些神经元

## Feature Map (a channel of a hidden layer)
一堆feature的集合，通过用同一个feature检测器，在同一个输入map的不同位置上，通过滑动窗口的方式（比如卷积）创建出来的。在同一个feature map 里的 features，有同样的感受野大小，查找同样的模式，但是是在不同的位置上。这种特性创造了 ConvNet 里的空间不变性
