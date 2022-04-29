1. 在解决的是什么问题？如何让模型很容易就变大，比如迅速增加 1000 倍 => 让单个层里面的参数显著增大，但计算量不增加
2. 为何成功，标志/准是什么？计算量没增加，但是模型容量/参数变大了，使用的是 Mixtures of experts，这种又叫高效、通用的条件计算(conditional computation)
3. 在前人基础上的关键创新是什么？ 1. 设计了 gating network，能够达到两个目标：实用到所有的 experts（重要性），把计算平均分配到experts上（负载均衡）。2. 提出两个技术来增加每个expert的batchsize，来最大化 GPU 的并行能力。 这几个方法可能naive：noisy-top-k gating, particular batching schemes, load-balancing loss, mixture-of-experts formalism
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？ 是这个领域的种子论文，告诉大家这个新赛道可以卷

balancing losses

efficiency metrics (TFLOPS/GPU) 

**sparsity of parameters** is among thhe most important modelling principles in machine learning, being used with great success in e.g. Lasso with the l1 penalty, in SVM with the hinge loss
and convnet by setting connections outside the receptive field to zero. 前两个不懂，后面这个好理解。就是参数里每次只有一部分是有用的，有点像 droppath，dropout 一样。本论文里的参数稀疏指不同 experts 里的参数没有互相连接。**计算路径**
也是每个训练样本定制的. 这是第一篇提出实用、可扩展、通用的神经网络


除了 MoE，还有其他可以用来迅速增加模型参数的方法：

1. Dense Layers: 比如  conv layer，每个参数对应每个算子，只需要O（1）的计算量，增加参数量会增加训练时间
2. Embedding Layers(搜广推) : 这个会非常稀疏。可以增加它的 feature space，也可以增加维度。我们认为 MoE 层比embedding layer 要更强大。 Embedding layer 会被网卡带宽所限制，因为要通过网络发送出去。而 MoE 只受GPU算力限制。
