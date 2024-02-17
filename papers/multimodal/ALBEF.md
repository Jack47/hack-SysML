1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

借用 ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision 里的图：

![](imgs/ViLT-4-categories-of-vision-language-model.png)

根据这4类，如何改进？
1. 模型结构：视觉特征远远大于文本特征，所以需要使用更大更强的视觉模型: 图像编码器要比视觉编码器要大；模态融合很重要:模态融合也必须要大。模型需要跟 C 类似；在多模态融合前，text 和 image embedding 的特征要 align，这样后面的多模态融合部分才比较好学。（本文两大贡献之一）
2. 如何训练？类似 CLIP 一样用 ITC(Image Text Contrastive) Loss 比较高效

![](imgs/illustration-of-ALBEF.png)

ALBEF 里用的是 BERT，但是只把前6层当作 text input，而后6层当作了多模态融合的编码器层。

为了提高从有噪音的 web 数据上学习的能力，提出了 momentum distillation，一种自训练(self-training)(一听到伪标签，就可以联想到是伪标签，从哪里来？)的方法，可以从 momentum model 产生的伪标签上学习。伪标签的概念来自 MoCo

Momentum Model 的 Moving average 的参数设置的很高(0.995)，所以更新很慢，产生的 feature 更稳定，用来产生伪标签，然后提供 Contrastive Loss 的更多 negatives。也就是说除了 VIT(image input) 和 BERT(Text input+多模态融合）参数之外，在 Momentum Model 里还有一份参数。

无论 LIM、ITM（image text matching)、还有 momentum distillation（动量蒸馏） 都是为同一个图像文本对生成不同的视角，也就是变相的 data augmentation

网上爬的大量图文对，里面噪音很大(noisy web data，比如 YFCC 100M 里筛完 nosiy 数据，只剩 1/6)，因为文本其实就是 alttext，它得对搜索引擎友好，所以并不是描述友好的。只是关键词的堆砌，并不是描述性的句子。

ITC Loss 是完全按照 MoCo 来的，Momentum Model 里产出的负样本会放到一个队列里，这些负样本并不需要梯度，所以不会占用很大。

利用 ITC Loss ，上面架构图里的 Image Encoder 和 Text Encoder 的前六层都可以训练了。ITM 就是一个二分类任务（图文是不是匹配的一个对？）。正样本还有点难度，负样本就很简单了，而且数量很多。所以预训练时很快精度就可以很高，训练再久就没意义了。所以常见做法是对负样本做一些限制：通过某种方式选择最难的那个负样本。 即最接近正样本的负样本(hard negatives)。这里为了算两个不同的 loss(ITC, Masked LM)，模型需要两次 forward( I,T 和 T、T'（masked T））。

在标准的 400M 的数据集上，只需要一台8卡 A100 训练3天，是多模态里非常亲民的。

noisy data 指文本和图像之间是弱关联：文本里含有跟图像不关联的关键字，或者图像有文本里没包含的实体。所以对 ITC 而言可能有个难的负样本是比正样本还精准的，而 MLM 的 loss （完形填空）里，可能有多个词都是符合的，所以可能存在比数据集里的 Ground Truth 描述的更好。所以一味惩罚负样本会让学习过程很艰难。那思路就是再找一些监督信息，把 one hot label 变成 multi hot label。才用了类似 google noisy student 和 MoCo 里自学习的思路，使用动量模型来生成 softmax score，而非 one hot label。在训练时，让 basemodel 的预测不光跟 one hot 的label 接近，还想让它跟动量模型输出的 pseudo targets 尽量接近。这样达到一个折衷，当 one hot label 里提供的信息不精确甚至是错误的，动量模型能提供改进。所以对于 ITC 和 MLM loss 都用的这种方法。

所以有5个 loss 函数：两个 ITC，两个 MLM，一个 ITM

因为此时ground truth 不再是一个值而是多值，所以从计算交叉熵变为了计算 KL divergence。

## 实验
ALBEF 无论在训练、推理、性能上都很亮眼(ViLT, UNITER)，而且还是开源的，算是22年多模态承上启下的工作。
