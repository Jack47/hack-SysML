讲的主要是如何构建一个海量、语义密集的数据集，并不是我关心的已有数据集里，添加原始数据。可以问问处理大数据集的编程、实践经验？

## Bamboo Dataset
它是一个大规模的视觉数据集，使用人机协作(synergy)来构建的有理解力的(comprehensive)标签系统:
a) 标签系统 BambooTX 从 WordNet 持续扩展而来。标签系统里的概念分为：通用视觉，非通用视觉，非视觉三个概念
b) 原始的通过关键字persion爬取的数据包含ID和非ID数据。ID 和 OOD 是啥意思？OOD 数据包含杂音，谐变量和语义变换数据。杂音数据代表不是物体。协变换。
数据包含人。但里面人太小了，annotator 无法lable。语义变换表示有其他语义物体（树）。为啥分为这三类？**OOD 修正(retification)** 通过过滤掉这些数据来
减缓 active learning在现实场景中的无效性。
c) Bamboo 主动收集了 65M 分类注释(annocation)和28M物体 bounding box annotation
knowledge base 用的哪个？ Wikidata

## 摘要
以前的数据集，要么标签是启发式的，要么注解是盲目的，没有区分性，让数据不高效而且无法扩展。主要挑战：
1. 系统地收集、annotate然后构建一个大数据集？

它是通过两方面特性，通过人机协作来构建的：
1. 标签系统：从24个公共数据集（19分类+5目标检测）和知识库(knowledge bases)里收集到170K新的类目，形成有层次结构的，拥有304k类目的标签系统。这个标签系统在我们设计的层次下，
很容易扩展，而且其中的概念进一步区分为“视觉”和“非视觉”
2. 主动标注：在370M label 系统爬取的原始图片基础上，只有informative的样本被选出来做人工标注，是通过我们的人机主动标注框架来做的。(human-machine active annotation是什么？机器选出最有信息量的，然后人标注)
我们发现rectifying OOD 样本对于主动学习在真实场景下高效工作非常重要。（active learning是什么？），类比 ImageNet22K 和 Objects 365，Bamboo的预训练模型能达到在分类上
超6.2，检测上超2.1的收益。由于标签系统和annotation pipeline的可扩展特性，bamboo可以持续增长并收益。

## 介绍
有两个设计哲学很关键：

1. 统一的、有表达力的标签系统：我们形成的统一标签系统有层次结构，包含304K类目。从24个开源数据集，以及从 knowledge base 里吸收了170K新类目（都是从 Wikidata 上来的？
而且我们发现标签系统里，只有视觉概念对模型训练有用。比如 ResNet-50 在 ImageNet22K 上没有包含非视觉概念情况下，性能在下游里比完整版本要好。基于这个发现，
我们在后续数据收集时，只保留了119K视觉类目

2. 主动数据注解(Active Annotation)Pipeline：
随即从没有标注的池子里拿图片来annotate是不高效的。所以研究员在探索主动学习（AL）算法来选择最有价值的样本来标注。然而，已有的AL方法大部分会在 OOD 的数据上失败，
因为他们在真实场景下不准确、不确定。由此我们提出新的 active data annotation pipeline：集成 OOD Rectfication，可以高效从海量野生数据池子里选取样本。减少模型的不确定性


主要特性：
* Comprehensive（理解力强？：65M 图片分类 annocation 以及28M object bounding box annotation，横跨 119K 视觉类别。lable和注解数据的规模都是已知开源数据里最大的
* Information-Dense：标签系统和注解数据实际目标是在覆盖面和信息有效性上都有保证。 
* 持续性：scale


想问问这套系统在里面发挥的作用，因为训练在做，数据系统也在构建，数据的压力是不是很大，因为得尽量超前准备好
他们的annotator pipieline 是怎么部署的？处理能力是不是有要求，比如128张图片/s

1. 通过 flickr 下载是靠哪个团队做的？
2. 人工标注主要是 Active Learning 之后，交给去做检测的标注吗？
