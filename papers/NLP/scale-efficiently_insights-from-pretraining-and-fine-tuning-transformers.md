1. 在解决的是什么问题？Transformer的一些未知问题，提供transformer 上预训练和 Finetune 方面的一些洞察
2. 为何成功，标志/准是什么？填补了 transformer finetune 方面和上游之间联动性问题
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？ 1. 对下游 finetune，除了模型大小，模型形状很重要。 2: scaling protocols operate differently at different compute regions 3: T5-base 和 T5-large 是低效的。它们重新设计的模型，提速40%，少了50%的参数。用的是它们提出的 effective improved scaling protocol.
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？


## 问题
1. 摘要里的 scaling protocols operate differently at different compute regions, scaling protocols 和 compute regions 各自是什么？
2. 

参考论文：

Scaling Laws for Neural Language Models: cross-entropy loss，和模型大小，数据集大小，训练消耗的算力之间是指数关系，而跟网络细节如网络宽度或深度之间关联不大（肯定是做了对比实验，发现影响很小）。但本论文说这些只考虑了 Pretrain，Finetune 情况下，Model 
Shape 也很重要


