本来是这个：https://yaofu.notion.site/An-Initial-Exploration-of-Theoretical-Support-for-Language-Model-Data-Engineering-Part-1-Pretraini-dc480d9bf7ff4659afd8c9fb738086eb#58a43041d42f4bb6bcad681b1cf056c5


** 数据优化**

预训练数据优化：找到最佳的 混合比例+数据格式+数据 curriculum，来让学习的速度最快

有监督的微调/instruction tuning data optimization: 找出最少的请求-响应对的结合，能让用户偏好的分布是最大的

结合预训练和finetuning，我们希望进一步找到方法来连接 pretraining 和 finetuning，进一步找到方法来连接他们来达到可预测的scaling。意思是给定预选连和精调的数据，模型结构和可训练的参数，可以在做实验之前得到结果

** 可预测的扩展**

# 2 使用 grokking 的速度来测量
Grokking 意味着在确定的训练步数上，模型突然学会了一个技能，从记忆迁移到了泛化

当输入是：预训练数据混合+模型规模+精调数据混合方法+训练算法，以便让 pretrain dev loss，下游性能，人类偏好都可以在做实验之前被预测

## 3.2 数据的 Curriculum

不能来源的数据带来不同的技能。以特定次序训练而非特定数据集上训练，可以有更快的训练速度:
(Skill-it!)[https://arxiv.org/abs/2307.14430] : 形式化了各种技能，证明了技能集存在。
## 3.3 混合比例

## 3.4 坑:模型规模
在 30B 以下的模型规模上，数据工程无法迁移到70B以上的模型里

## 4.2 Aggregating single-skill grokking to scaling law and emergent abilities

上述图表里，可以看到：

1. 单个技能(quantum)曲线通常展示了 phase-change shapes
2. 模型学习不同技能的速度不一样
3. 把多种技能聚合到一起，得到图里丝滑的 log-linear 的 loss curve




这里有个论文：[The Quantization Model of Neural Scaling](https://arxiv.org/abs/2303.13506)

Training Compute-Optimal Large Language Models: 发现模型大小和过的token，应该同步扩

里面最有意思的是 5：

# 5.无损压缩，Kolmogorov 复杂度和生成的过程：

5.1随机数生成器的例子

5.2Kolmogorov 复杂度

OpenAI 说在 GPT-4 开发过程中，在实验之前就预测了 HumanEval 上的性能。我们认为这种预测可以被统一到一种已知的理论上：能用来预测下游的任务
