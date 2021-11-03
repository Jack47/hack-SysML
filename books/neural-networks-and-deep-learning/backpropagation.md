
反向传播成核心是损失函数 C 对应到权重w 或偏置 b 的偏导。这个偏导代表了修改权重后，损失变化的大小/剧烈程度

a^l = theta(w^la^(l-1)+b^l) = theta(z^l)

其中 theta 代表对矩阵里每个元素做操作f

而 z^l 叫做 weighted input

关于损失函数的两个假设：

1. C = 1/n*sum(Cx) : 损失函数是对每个训练例子的平均损失
2. C 是关于最后一层的函数： C = C(a^L)

用平方损失函数举例：

elementwise product: python 里的 矩阵 x*y 就是这种乘法。而 x@y 是矩阵乘法

## 反向传播背后的四个等式

反向传播的目的是衡量修改权重或偏置后，损失函数如何变化。这不就是导数的定义么，那么 deltaC = f(Zj)'*delta(Zj)

本质是对这两者求偏导。为了求偏导，先求出损失函数对最后一层的偏导，然后把这个损失再传播到上一层

这个也是导数的定义：f'(x) = delta(y)/delta(x)，衡量的是x变化时，y的变化情况。导数就是在这个点上的变化率。导数的本质是通过极限的概念对函数进行局部的线性逼近

先定义每一层的关联损失为 , 然后把这些联系到实际的权重和偏置

主要是计算两个：

1. l 层的损失关联函数
2. 损失函数的梯度



核心还是计算导数，然后推导出 l+1 与 l 之间的关系，依靠输入输出关系，求导，然后就有了输出和输入即下一级导数之间的关系

pytorch 会帮我们去计算两个数的乘积：grad_out = delta * theta'


## PyTorch 里 backward

```
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.saved_for_backward(input)
        return 5*input**3 - 3*input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output*15*input**2-3
```
从上面可见，backward(ctx, grad_output) 里，输入的是损失对于输出的梯度，而 `grad_output*f'(x)` 就是当前层的梯度：损失对于当前输入层的梯度。所以这里提到的梯度，就是上面的 feil

而其中 backward 里的 grad_output = (w^l)^T*fei(l+1)


## 问题
1. pytorch里实现 backward(input, grad_out, output), 这几个参数各自是书里的哪部分？backward 其实就是求出对于 weight 和 bias 的权重？
2. gamma1*X，这个步骤的 backward 是什么？是 X ？

## TODO
把书里的部分图片搞到这里来
