
## Session 1: 

### The Need for Speed: From Electric Supercars to Cloud Bursting for Design

Dean Drako 会比较与电子超级汽车相比的对速度追求的行业驱动力。深度探讨4个核心要素，让设计和验证组加速部署来让创新产品的交付加速。


## Session 2: Placement, Clock Tree Synthesis(合成), and Optimization

### RTL-MP: 面向实用，人类质量的芯片规划和微放置（Toward Practical, Human-Quality Chip Planning and Macro Placement）

典型的 RTL-to-GDSII 流程里，布局规划对达到结果质量发挥非常重要的作用。一个好的布局规划通常需要前端设计者--负责RTL的功能，和后端物理设计工程师进行交互。而
macro-dominated 设计(尤其是有自动产生 RTL 的机器学习加速器)的复杂性增加，让布局规划更加挑战和耗时。本文提出了 RTL-MP，是创新的macro placer，使用了 RTL 信息，尝试“模仿”
RTL前端设计者和后端物理设计师之间的交互，来产生人类质量的布局设计。通过利用逻辑层次和处理基于连接特性的逻辑模块，RTL-MP捕获了在 RTL 里继承的数据流，使用数据流信息来指导 macro placement。
我们也使用自动调优来优化超参数。

### Kernel Mapping Techniques for Deep Learning Neural Network Accelerators
深度学习应用是计算密集和天然并行的；这刺激了专门为这种负载优化的新的处理器架构的发展。本文里，考虑了深度学习神经网络和更传统电路结构上的差异--高亮了这如何影响把神经网络计算内核映射到可用的硬件上。
我们通过动态规划来呈现一个高效的映射，也有一个方法来建立性能边界。也提出了一个架构方法来延长硬件加速器的实际寿命，让多种异构处理器可以集成为一个高性能系统。

## Session 3: Design Flow
