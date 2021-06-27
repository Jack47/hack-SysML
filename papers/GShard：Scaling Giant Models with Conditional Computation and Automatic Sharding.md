
实现细粒度的模型并行。

![](https://pic4.zhimg.com/80/v2-8e342553ff46dd88d16e265ed530b5b7_720w.jpg)

主要贡献是提供了一套原语，允许高层开发者通过简单的方式指导代码生成（编译期）生成对应的模型并行的代码。

## 摘要
GShard 由一套轻量级注解 API 组成，是 XLA 编译器的扩展。它提供优雅的方式来以对现有模型最小的改动去表达大量并行计算的模式。GShard 让我们可以用自动sharding的方式，把Sparsely-Gated Mixture-of-Experts 里的多语言神经机器翻译 Transformer 模型扩攒到超过6k亿参数