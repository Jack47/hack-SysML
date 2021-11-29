1. 在解决的是什么问题？验证了市面上量化算法的**可复现性**和**可部署性**，制定了一套 benchmark 的方法
2. 为何成功，标志/准是什么？讲清楚了学术界的有些量化算法为何无法部署，讲清楚了学术界和工业界的差异
3. 在前人基础上的关键创新是什么？定义了几个benchmark 的维度，发现了一些有趣的洞察
4. 关键结果有哪些？除了 benchmark，还讲了 BatchNorm 和图结构。设计了一套标准/唯一的 pipeline 来做测试可复现性
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？后续学术界搞出来的量化方法，得考虑可部署性了

## 1. 介绍
大模型能在云上进行计算，然而这些模型部署到边缘侧设备时，由于资源受限、延迟、内存消耗和功率开销都受限，所以非常难。为了解决这些问题，为了加速边缘侧的推理，涌现了很多技术：模型量化，剪枝，网络蒸馏，轻量级网络设计，权重矩阵分解。

推理时量化目标是把（几乎）连续的 32 bit 浮点数映射到离散的低比特整数。这样让边缘侧的设备可以用整数运算单元来计算。

发现：

1. 训练的超参数可以很大程度影响量化网络的性能
2. 学术研究的论文没有关注他们的算法是否能在实际的硬件上运行。所以报告的性能不可靠:
 a：硬件会把 BN 和 卷积合并到一起。但是大部分研究的论文都是保留 BN。
 b: 另外，论文里只关注卷积层的输入和权重参数的量化。而实际部署时整个计算图都应该被量化(这个是为啥？也不是绝对的吧)
 c: 算法鲁棒性：per-tensor量化的算法被应用到 per-channel 的量化上，会怎么样？

## 2. MQBench： Towards Reproducible Quantization
硬件感知的量化器：只本论文里只考虑对称量化，因为非对称量化（zero point 不是固定死到0的，可以调整）需要硬件支持。

量化器分为这几种：

1. 对称和非对称
2. Per-tensor ， Per-channel
3. FP32 scale 或 POT(Power of Two) scale

## 3. Towards Deployable Quantization
介绍了可部署的一些要求

## TODO
1. 看看轻量网络设计：MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf
2. 看看网络蒸馏：A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
## 问题：
1. 剪枝就是砍 channel 数量？
2. MQBench 是不是得提供硬件平台来实际部署，才能验证算法的可部署性？
