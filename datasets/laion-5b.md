LAION-5B 里总共有 五千八百五十亿的 CLIP-filtered image-text pairs, 比 LAION-400M 这个之前开源的最大图片-文本对要大 14倍。

![](imgs/laion-5b.md)

大的图片-文本对模型比如 ALIGN，BASIC，Turing Bletchly, FLORENCE & GLIDE 等都展现出了比原来的旗舰模型比如 CLIP 和 DALL-E 更好的性能。大部分是在十亿级别的图片文本对上训练的，但是这个大小的数据集，至今都没有开源的。为了解决这个问题，我们呈现了 LAION 5B，一个大规模的研究数据集，包含 5,85B 的 CLIP 过滤过的 image-text 对。2,3B 包含英语，2,2B 采样自其他100种语言，1B 采样自不知道具体语言的文本。此外，我们提供了几种最近的邻居索引，一个 web ui 来探索&创建子集，同时包含水印的检测的分数和 NSFW。我们也发布了在 LAION-400M 上完整复现 clip 训练的方法：[open clip](https://github.com/mlfoundations/open_clip)

## 数据集目的和内容警告的免责声明
数据集背后的动机是平民化大规模多模态模型训练方面的研究和试验。我们的建议是只用于研究目的。注意这个大规模数据集是裸数据集(uncurated)

## 介绍
自从 2021 年一月 CLIP & DALL-E 的发布，有很多类似的大规模、多模态的语言-视觉模型已经被一些大机构训练出来。比如 FLORENCE，Turing Bletchley，ALIGN & BASIC 都证明了在创新的数据集上面，即使没有每个样本粒度的标签，也可以有很强的转移能力，当训练数据量变大，会持续提升，遵守了之前研究成果里的 scaling laws。这些模型需要十亿级别的图片文本对来达到高性能，然而，之前都没有十亿级别的图片文本数据集开源。了解决这个问题，我们发布了 LAION-5B，一个 CLIP 过滤过的 5,85 billion 的高质量图文对，对应的 CLIP ViT-L/14 的 embeddings，kNN-索引，web ui，NSFW- 和水印检测分数和工具。我们描述了创建数据集的过程，证明了成功训练 DALL-E 的架构。有了充足的大规模数据集，让更广泛的社区可以开启研究多模态的语言视觉模型

## 下载数据
在 LAION-5B 项目上，发布了如下的包：

laion2B-en

laion2B-multi

laion1B-nolang

数据集可以通过[im2dataset](https://github.com/rom1504/img2dataset) 这个python 程序来下载: Easily turn large sets of image urls to an image dataset. Can download, resize and package 100M urls in 20h on one machine. （240 TB的 384，80TB 的 224）

对于训练场景下的使用，推荐阅读[训练用法指导](https://github.com/rom1504/laion-prepro/blob/main/laion5B/usage_guide/preparing_data_for_training.md)

特定的，我们发布的这个数据：

* 5.85 billion 的图片url对，对应的 metadata 在 laion2B-en，laion2B-multi, laion1B-nolang(800GB)
* knn index，可以容许快速检索 laion5B 数据集(1.6TB)

