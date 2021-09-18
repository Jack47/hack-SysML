1. 在解决的是什么问题？传统 ImageNet-21K 由于 label 不是相互排斥，只有一个标签等原因，不怎么流行。反而 ImageNet 1K 用的多，成为 pretraining 的主要数据集。我们的方法，可以让 Imagenet 21K 预训练在众多数据集和任务上，能极大收益。
2. 为何成功，标志/准是什么？预训练后，下游任务精度高。相当于实现了 efficient top-quality pretraining on ImageNet-21 K
3. 在前人基础上的关键创新是什么？semantic softmax, 数据预处理，WordNet semantic tree 多标签，
4. 关键结果有哪些？在多个下游任务上和不同大小的模型上都表现优异
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？大家都可以用它这个 ImageNet-21K Processed.


## 启发
1. 代码里有 hierarchical semantic tree 的加载源码
2. 里面用到了知识蒸馏：有[老师和学生](https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/semantic/semantics.py#L104)
3. 它提供了好几个版本的 ImageNet 21 K 预处理过的数据集

