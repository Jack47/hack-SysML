1. 在解决的是什么问题？为什么 Masked AutoEncoder在NLP上表现那么好，而在 CV 里表现不佳？
2. 为何成功，标志/准是什么？可以在CV上自监督，而且需要的数据量小，在下游任务上比监督任务要好
3. 在前人基础上的关键创新是什么？设计了非对称的 encoder-decoder 结构，mask 输入图片到 75%
4. 关键结果有哪些？挖掘出 NLP 和 CV 的不同点：图片是光学记录的影像，没有语义上的分解到光学类比的文字
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 介绍
NLP由于可以自监督(输入句子里遮住一些单词，然后让机器预测这些单词)，所以参数量可以搞很大。而这些方法都是基于自动回归的。在 CV 里虽然也有类似研究，但是落后于 NLP。我们想：
是什么让masked autoencoding在视觉和语言上表现不同？答案或许是：

1. 直到最近的 ViT 之前，两者架构是不一样的。之前都是 卷积网络。不好把 masked token 这种 indicators 或 positional embeddinggs 集成到 CNN 里
2. 信息密度不同，语言是人发明的信号，有高度语义和信息密度。而图片相反，是自然信号，有高度的空间冗余--例如其中丢失的一块可以依靠邻居来复原，不需要太多高度理解的部分对象和场景信息。所以我们 mask 的比例很高
3. autoencoder 里的 decoder ，可以把缺失的表示映射回输入，在构建文字和图像之间是非常重要的一环。