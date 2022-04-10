
## 摘要
预训练通用目的语言模型可以在多种自然语言处理领域，通过zero-shot、few-shot和finetune等方法，适配到下游任务，达到领先水平。由于他们的成功，模型就越来越大，需要高性能的硬件、软件和算法技术来训练如此大的模型。

## 问题
1. 文中提到的 zero-shot、few-shot 是什么方法？
通过微软和NV的联合，我们训练出了最大的单体 transformer的语言模型，Megatron-Turing NLG 530B。本文论文里，我们首先聚焦在基础设施以及3D并行的方法。之后详细介绍训练过程，语料的设计，数据积累技术，
我们相信这些是模型成功的关键技术。最终，讨论了多种评估方法，还有其他有趣的观察和MT-NLP展示的新特性。我们证明MT-NLG 在几个 NLP benchmarks 上超过了zero，one，few-shot 训练精度。

## TODO
1. Language models are few-shot learners.
