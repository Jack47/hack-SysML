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

## torch.one_hot
总结：扩展到更高维度，但是高纬里只有一个位置是1，即原来低纬度里所指的位置，比如 x = [i][j] -> [i][j][x]=1，其他[i][j][y] 都是0。而且这个函数是无法指定维度的
看下面的例子，相当于把 index 变成了 mask

```
num_experts = 32
expert_index = torch.randn((num_cores, tokens_per_core])
expert_mask = torch.one_hot(expert_index, num_classes=num_experts) # -> [num_cores, tokens_per_core, num_experts], 其中只有原来 [n, t] == 1 的位置才会是1，其他 [, , x] 都是0

import torch.nn.functional as F
l = torch.tensor([1,2,0])
x = F.one_hot(l, num_classes=3)
x
tensor([[0, 1, 0],
        [0,0,1],
        [1,0,0]))
```

## torch.cumsum
总结：维度不变，逐渐把值累加起来
没太理解的是维度，比如 0，它是在列上逐行求和。

## torch.mean
也叫 reduce_mean，没有指定维度的情况下，取的是所有元素的平均。

总结：沿着 dim 在另外一个维度上求和，所以指定的维度会消失。即这个 dim 改名叫 reduced_dim 好一些

```
>>> l = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float)
>>> l
tensor([[1., 2., 3.],
        [4., 5., 6.]])
>>> torch.mean(l)
tensor(3.5000)
>>> torch.mean(l, dim=0) # 指定的行维度消失
tensor([2.5000, 3.5000, 4.5000])
>>> torch.mean(l, dim=1) # 指定的列维度消失，但没 keepdim
tensor([2., 5.])
>>> torch.mean(l, dim=1,keepdim=True)
tensor([[2.],
        [5.]])
```
