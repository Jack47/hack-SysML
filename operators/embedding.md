它是一个简单的查表过程，里面存储了 num\_embeddings 个元素，即字典的大小。embedding\_dim  是每个 embedding 向量的大小 [1, embedding\_dim]。本质是通过元素的索引，查到这个元素背后对应的这一行，作为输出。

```
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)
```

它里面的权重为 [num\_embeddings, embedding\_dim] 大小

它和 Linear 的区别：Embedding 实现的功能 Linear 也能实现，但是 Embedding 是查表，是把离散的值映射到连续的向量空间，这样对于NLP任务很有用，可以捕捉单词之间语义等关系，而 Linear 是矩阵乘。

问题：embedding 里的权重是怎么学习的？即查表的梯度怎么算
