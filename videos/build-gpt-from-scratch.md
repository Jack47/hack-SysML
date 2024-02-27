## Self Attention
先实现了一版用均等的权重去做自己和之前每个token的和。后面想演化成权重不是均等的，而是数据决定的

Q: what i am looking for
K: what do i contain

```
x = torch.rand(B,T,C)
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)
wei = q@k.transpose(-2, -1) # (B,T,head_size) @(B,head_size,T) -> (B,T, T)
tri = torch.tril(torch.ones(T,T)) # todo: torch.tril
wei = wei.masked_fill(tri==0, float('-inf')) # 这一步提前做能提高效率吗？
wei = F.softmax(wei, dim=-1) # 这样就学到了关联关系
v = value(x) 
out = wei @ value

```

value # private information on this token, get aggregated on 

Q@K 之后就是那个wei # affinities and connections

note1: attention as communication mechanism. 每个 token 都有前面的 token 指向它

note2: attention 没有空间的概念，它只作用于一个集合上，所以需要 positonal encode

note3: no communication across batch dimension

note4: encoder block 里是没有 tril 的，所以未来的 token 可以和之前的 token 之前有关系。上面的代码就是 decoder 模式

cross attention：key 和 value 来自其他地方(比如encoder），只有 q 是来自 x。而 self attention 里 key 和 value 都来自 x

note5: 为什么实现时 q@k/sqr_root(head_size) ? 是为了控制结果的 variance 不要太大，否则过完 softmax（它自身特性）值会偏大


## 问题
1. 那些用到的函数，以及对数据做的变换
2. 51:31 这里没太看懂 a 为什么就变成平均了？

## 常见函数
### torch.randint
```
torch.randint(len-batch\_size, (batch\_size,)) # (batch\_size,) 是个 tuple，用来表示要生成的 tensor 的 shape
```

### 形状不同的张量，做算数运算时可以广播到相同的形状
```
tok_emb = self.token_embedding_table(idx) # (B, T, C)
pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
x = tok_emb + pos_emb # 两个张量的形状不一致，此时会广播
## 等价于
# pos_emb_expand = pos_emb.unsqueeze(0) # 在第1纬增加大小为1的新纬度，=> [1, T, C]
# x = tok_emb = pos_emb_expand
```

### cross_entropy
两个入参：input\_logits, target。target 可以跟 input 纬度相同(代表class probabilities)，也可以少一纬度(代表实际类别，其他类别概率为0）

默认会用 mean 做 reduction，所以结果是一个标量。reduce=none 之后就可以拿到平均前的，比如 [B, T] 大小？
```
logits = logits.view(B*T, C) # [B, T, C] -> [B*T, C]
targets = targets.view(B*T) # [B, T] -> [B*T]
loss = F.cross_entropy(logits, targets) # [B*T]
```
举例：
```
>>> # Example of target with class indices
>>> input = torch.randn(3, 5, requires_grad=True) # [3, 5]
>>> target = torch.randint(5, (3,), dtype=torch.int64) # [3]
>>> loss = F.cross_entropy(input, target) # 被平均之后的标量
>>> loss.backward()
>>>
>>> # Example of target with class probabilities
>>> input = torch.randn(3, 5, requires_grad=True) # [3, 5]
>>> target = torch.randn(3, 5).softmax(dim=1) # [3, 5] # 对第二纬统计概率
>>> loss = F.cross_entropy(input, target)
>>> loss.backward()
```

### torch.multinomial
多分类概率分布(比如 vocab_size）中进行多项式(multinomial)抽样
input 形状为 (N, C)的张量，其中 N 是样本数量，C 是类别，里面包含了每个样本的概率分布

```
logits = logits[:, -1, :] # (B, T, C) -> (B, C) # 只保留了第二纬中最后一列
probs = F.softmax(logits, dim=-1) # (B, C)
idx_next = torch.multinomial(Probs, num_samples=1) # (B, 1)
idx = torch.cat((idx, idx_next), dim=1) # (B, T) + (B,1) -> (B, T+1)
```
