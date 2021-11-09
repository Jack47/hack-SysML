## relu
```
def ReLU(x):
    return x * (x>0)
    
def dReLU(x):
    return 1 * (x>0) # 意思是 ReLU 并不改变最终结果，只不过是把一些 neuron 给短路掉了
```

## gelu
backward 和 forward 比较复杂

```
