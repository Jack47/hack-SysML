下图里看出同样的话，英语需要的 token 数量比韩语要少。

densify python，so we can attend more。


## 常见函数

### zip
将多个可迭代对象（列表、元组、字典等）的元素打包成一个个元祖。zip 会并行地从每个可迭代对象中取出相应位置的元素，直到最长可迭代对象被完全迭代。

```
for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
return counts
```

## 问题
1. tokenizer 的 train 过程，也需要看 loss？还是说只是限定需要走多少次？
2. 