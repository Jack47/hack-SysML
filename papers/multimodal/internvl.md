1. 把 vision foundation model scale 到了 6B 参数
2. 渐进地训练：现在大规模网络数据集上进行对比学习，然后再在高质量数据集上做生成式学习

Language Middleware：QLLaMA：基于多语言的 llama，然后加上96个可学习的 query 和 cross attention

![](imgs/intern-vl-training-strategy.png)
