## 常见函数
### View
不改变底层存储的情况下，改变 tensor 的 shape，比如三维变二维，或者反之

```
embcat = emb.view(emb.shape[0], -1) # -1 指自动计算这个纬度（第二维）
embcat.view(emb.shape[0], N, C)  # view 的参数就是指定有哪几个纬度，各自是多少
```

### torch.mean
计算某个纬度上的均值
```
torch.mean(input_tensor, dim=None, keepdim=False) # dim=None 指计算所有元素的均值
hpreact.mean(0, keepdim=True) # 在 B 这个纬度上计算
hpreact.std(0, keepdim=True)
```