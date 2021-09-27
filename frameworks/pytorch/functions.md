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

torch.tensor.expand():  Returns a new view of the self tensor with singleton dimensions expanded to a larger size. Passing -1 as the size for a dimension means not chaning the size of that dimension. It doesn't allocate new memory.

nn.Identity: A placeholder identity operator that is argument-insensitive. m = nn.Identity(54). output = m(input), 此时 output 就是 input

torch.cat(tensors, dim=0, ) : 在指定的维度上，对众多 tensors 进行拼接，是 torch.split() 的反向操作

