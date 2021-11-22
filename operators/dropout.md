
## dropout 实现
```
mask = (torch.rand(X.shape) > dropout).float() # 1 or 0, 对于随机出来的数字，大于  dropout 概率的才保留，否则乘积后为0
keep_prob = 1.0-dropout_prob
return mask*X / keep_prob
```
假设总共有 n 个 neurons, p = 1-dropout 是保留的比例。那么原本期望是 x，用了 dropout 之后，期望是 `p*x`，所以为了保证期望不变，需要 mask*X / p

### Backprop
由于 dropout 里并没有要学习的参数，而且有些 neuron 已经变成了0，所以不需要反向传播这些 neuron 的梯度。所以：

```
backward(input, grad_out, output)
 mask = ctx.saved
 return grad_out*mask/keep_prob # 让梯度通过不为0的 neuron 传递回去
```

## droppath 实现
```
shape = (x.shape[0], ) + (1, )*(x.ndim-1) # work with diff dim tensors
random_tensor = keep_prop+torch.rand(shape)
mask = random_tensor.floor_() # binarize 意思是当前 neuron 是否保留
return x.div(keep_prop)*mask
```

## 差异
DropPath 是丢掉 batch 里整个采样(batch 里的某一个sample 直接丢弃掉)

Dropout 是丢掉输入 tensor 里的某些元素

## 问题
1. 没看出上述实现的差异点: mask 都是0或者1。都是 X.div(keep_prop)*mask
2. 为什么 dropout 里要除以 1/p? (1-dropout)
3. 是否当droppath 里某个 batch 被丢弃掉，它就永远不能参与运算了？ 错，只是当前这次

##  参考资料
 1. [Droppath in Timm seems like a dropout](https://stackoverflow.com/questions/69175642/droppath-in-timm-seems-like-a-dropout)
