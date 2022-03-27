## torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) bool
用于逐个元素检查在容忍范围内，两个数组是否相等

会检查 input 和 other 是否满足：｜input-other | <= atol + rtol * |other|

atol: absolute tolerance. Default: 1e-08

rtol: relative tolerance. Default: 1e-05

## Construct a Sequential Model
```
from torch import nn
m == nn.Sequential(nn.Linear(2,2), nn.ReLU(), nn.Sequential(nn.Sigmoid(), nn.ReLU()))
```

## Recursively iterate over modules
m.modules() : 会嵌套打印出所有级别上的模型

m.children 不会进入第二级的 module 里

m.named_modules()

用list举例就是：

```
a=[1,2,[3,4]]
```
children返回

```
1,2，[3，4]
```
modules返回

```
[1,2,[3,4]], 1, 2, [3,4], 3, 4

```

### torch.tensor.expand()
Returns a new view of the self tensor with singleton dimensions expanded to a larger size. Passing -1 as the size for a dimension means not chaning the size of that dimension. It doesn't allocate new memory.

### nn.Identity
A placeholder identity operator that is argument-insensitive. m = nn.Identity(54). output = m(input), 此时 output 就是 input

### torch.cat(tensors, dim=0, ) 

在指定的维度上，对众多 tensors 进行拼接，是 torch.split() 的反向操作




## torch.Tensor.register_hook
可以挂(唯一) **一个** hook，当这个 tensor 的梯度被计算时使用

## torch.nn.Module.hooks

Forward hooks, Backward hooks

## torch.view(shape) -> Tensor
在原始tensor数据不改变的条件下，返回新的tensor。要求新tensor的纬度是原始纬度的子空间，否则需要用 reshape() 来进行维度变换，此时得到的 tensor 不与原始 tensor 共享内存

## stride 属性
tensor的 stride 步长，代表从索引中的一个维度跨到下一个维度的跨度
