1. 在解决的是什么问题？虽然之前的 MoE 和 Switch Transfromer 已经是高效的方法了，但是由于训练不稳定，fine-tuning 时质量并不高
2. 为何成功，标志/准是什么？解决了上面的训练不稳定的问题，在 269B 参数上训练，计算代价等效于 32B 的 dense。第一次 sparse 模型在迁移学习上有了 SOTA 的性能。
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？找到修复稳定性，不损害质量的方法(就是 z loss？）
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？


## 1 介绍

## 2 背景

## 3 让 Sparse Models 的训练稳定下来
我们所说的训练过程不稳定，指的是训练过程中 loss 的爆炸。有以下几种：

### 3.1 删掉  Multiplicative Interactions 后的稳定性和质量上的折衷
更多 multiplicative components： GEGLU，RMS norm，稳定性差，但是性能更强

### 3.2 增加 Noise 后的稳定性和质量上的折衷
这个是怎么加的？包含两部分：input jitter(10^-2) 和 dropout(0.1)

### 3.3 限制激活值和梯度后的稳定性和质量上的折衷
常用的最成功的让 nn 训练更稳定的方法是激活值和梯度上的限制。最流行的方法是 clipping 梯度 norms 来缓解爆炸的梯度。

本工作里，使用 Adafactor optimizer，因为内存高效（尽管最近有一个 8-bit 的优化器）。没用 gradient clipping，Adafactor 使用了 update clipping，对权重的改动限制在小于特定的 norm。

另外研究了在 router 的 logits 上加限制。router 计算了 experts 上 float32 精度上的概率分布。然而，大规模上，这样训练不够提供可靠的训练。为了修复，我们提出了 router z-loss。而且也实验了把 z-loss 加到 attention logits 里，显著提高了模型的稳定性。

使用 hyperparameter sweep 之后，得到 coefficient 权重为 0.001 时更好

3.4 选择合适的精度：在效率和稳定性上的折衷

激活值和计算时是 bf16，allreduce 用的 float32

数字越大，roundoff 的误差越大。目前看到参数上的变化是非常小的

Sparse expert 模型对舍入误差更加敏感，因为他们有更多的指数函数，因为 router 上有更多的指数函数

Clipping logits 发生在 roundoff 误差之后，导致更大的非连续性。clipping 也可以认为是一种舍入误差；因此，z-loss 自然鼓励模型产出小 logits 因此更加准确。由于这些动态性，我们确保所有的指数 tensor 都是 float32 运算的（见9）
## 4 Fine-Tuning Sparse Models 时的性能
### 4.1 一个泛化问题的假设
sparse 模型更容易过拟合

4.2 只 FT 参数的一个子集合来提高泛化性
大概 2/3 是性能相当的，更少了就弱了

4.3 Sparse 和 Dense Model 需要不同的 FT
更小的 batch size 和更高的学习率（4.1 里提到更容易过拟合）


4.4 FT 期间，Sparse Models 对 Dropped Tokens 更鲁棒
发现 load balance losses (Aux Loss) 提高了 Fine-tuning 
Token dropping 有点像正则化。
4.5 在 FT 期间插入哨兵 Tokens

## 5 设计 Sparse Models

5.1 expert 数量

在 TPU 上，推荐一个核上顶多1个 expert，来确保计算-内存比例比较搞，减少 evaluation 和推理所用的 cores
5.2 选择 CF 和 Routing 算法

## 6 实验结果

## 7 跟踪模型里的 tokens
Encoder Experts 展示出 specialization

Decoder Experts 缺乏 specialization

多语言上 Expert 的 specialization，但不是特定语言

A Token 的 Load Balance 描述

B Router Z-Loss 训练的动态性

C 架构上的提高改动

D 低 Capacity Factors 上的 Batch Prioritized Routing

E Pre-Training Dataset Details

F Full Fine-tuning Sensitivity Data

G 最优地设置 Routing 的阈值

I 分布式模式下的通信开销

J 负向的实验


