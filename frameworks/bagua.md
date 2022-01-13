想看如下几个问题：

1. hierarichical 部分的代码
## 2. tcp 优化部分

## 3. fused flatten optimizer
代码中把这个特性叫做 gradient as bucket view(do flatten)，更能精准表达含义。这个并不能减少那次拷贝，但是会增强程序的局部性。跟 LS 里把所有 param 和 grad 放到一个超大 tensor 里类似的机制

## 4. 第一个 iteration 里体现的 profile 阶段

