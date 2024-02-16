##

codex 的数据集上性能差不多。太短的代码就类似把任务描述翻译为代码，所以 codex 解决的问题更容易。编程竞赛就两个：

1. ICPC: International Collegiate Programming Competition:本科生
2. IOI: International Olympiad in Informatics: 高中生

## 2 evaluate
n@k 跟 pass@k 类似

## 3 数据集
两个(跟 codex 类似）：
1. 预训练
2. 微调的

不仅用了 python，还用了主流的语言（C++、Java，跟后面数据集相关）。数据集比一年前的 codex 大了5倍-> 715GB

因为目的是竞赛，所以使用了 CodeContests 数据集，是来自 Codeforces 上的数据，有用户提交了自己的正确答案（C++、Python、Java）

 微调的数据集上，还判断了 False Positive 和 Slow Rate
 
## 4 模型结构
用的 encoder-decoder 这种标准的 transformer 结构。

解码器个数是编码器的6倍

Query 上有多个 head，k和v 上只有一个。

非对称的结构：encoder 里是 1536 tokens，而 decoder 是 768 tokens，因为描述文本总是比代码长度要长一些。

预算练时是比较标准的

### 4.3 Fine-tuning


GOLD：离线的 RL 算法

### 4.4 Large Scale Sampling
为了多样性，才用了：

1. 生成的代码，一半用 Python，一半 C++
2. 随机化问题的 tags 和 rating
3. 使用相对高的采样温度

### 4.5 Filtering
生成了上述大量的解，如何过滤出靠谱的？

1. 用描述里的样例，就可以过滤掉99%的样例
2. 聚类来筛选，找出topk 大的聚类，从里面找样例去提交

为了采样速度的考虑（采样越高分数越好），所以用了本文里的架构，而非纯解码器和标准的多头注意力。

## copying from training data

### 
