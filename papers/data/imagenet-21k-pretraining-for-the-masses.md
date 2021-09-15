1. 在解决的是什么问题？传统 ImageNet-21K 由于 label 不是相互排斥，只有一个标签等原因，不怎么流行。反而 ImageNet 1K 用的多，成为 pretraining 的主要数据集。我们的方法，可以让 Imagenet 21K 预训练在众多数据集和任务上，能极大收益。
2. 为何成功，标志/准是什么？预训练后，下游任务精度高。相当于实现了 efficient top-quality pretraining on ImageNet-21 K
3. 在前人基础上的关键创新是什么？semantic softmax, 数据预处理，WordNet semantic tree 多标签，
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

