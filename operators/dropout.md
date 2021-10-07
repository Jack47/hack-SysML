
## dropout 实现
```
mask = (torch.rand(X.shape) > dropout).float() # 1 or 0, 对于随机出来的数字，大于  dropout 概率的才保留，否则乘积后为0
return mask*X / (1.0-dropout)
```

## droppath 实现
```
random_tensor = keep_prop+torch.rand(shape)
mask = random_tensor.floor_() # binarize
return x.div(keep_prop)*mask
```

## 差异
DropPath 是丢掉 batch 里整个采样

Dropout 是丢掉输入 tensor 里的某些元素
## 问题
1. 没看出上述实现的差异点: mask 都是0或者1。都是 X.div(keep_prop)*mask

##  参考资料
 1. [Droppath in Timm seems like a dropout](https://stackoverflow.com/questions/69175642/droppath-in-timm-seems-like-a-dropout)
