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
