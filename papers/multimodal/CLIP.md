1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？


没看懂的是为什么 clip github 首页上的例子里，两个矩阵相乘之后，就可以算相似度呢？而且输出的就是单词了，而不是字母或者没有含义的

```

similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)
```

## 摘要
我们证明简单的预训练：预测图片上标题就可以是高效并且可扩展的方式来学会 SOTA 的 image 表征，在 4亿图文对上就可以做。预训练之后，就可以使用自研语言来指代学到的视觉概念，产生的 zero-shot 的迁移到下游任务上。们在30个不同的已有的 cv 数据集，比如 OCR，视频里的动作识别，地理位置，其他各种细粒度的物体分类。这个模型很容易迁移到更多类任务上

任务无关的目标，比如自回归(autoregressive)(预测下一个）和语言掩码模型（完形填空）方式可以在计算，模型大小，数据方面scale，能力能稳定提升。这种标准的“文本到文本”的方式是任务无关的架构，可以 zero-shot 迁移到下游数据集上，移除了需要特定的输出头或者数据集特定定制。旗舰系统就是 GPT-3，它可以用在非常多的系统上，只需要少量或者无须特定领域的数据集。

2017 年时已经有人做了类似的工作，发现有 zero-shot 能力，但是当时没有 transformer，也没有 CLIP 这里几个月的算力支持，也没有 CLIP 里 400M 的数据集。之后也有人做了文本弱监督相关的工作，但是作者认为由于训练的时间和数据集规模没有上去，所以效果并不好。

CLIP 是一种方法，里面使用了从 ResNet 到 VIT 的各种以及不同大小的模型，发现模型增大，能力就变强

更加稳健：当效果和有监督任务持平时，CLIP 的泛化能力更加强（香蕉的那个例子，对抗样本里有监督任务在瞎猜）

MOCO 或者 MAE 里面都是单纯的视觉特征，而一旦把语言和图像结合到一起就有了多模态的能力，就有 zero-shot 迁移能力

WIT：WebImageText 数据集

![](./imgs/clip-summary.png)
## 2 方法

预测换为对比目标，效率就提升上去了。OpenAI 的 GPT 都是预测类的方法，唯独 CLIP 为了训练效率考量，使用的是对比学习的方法。

约束放宽，训练后的效果就更高了

这里应该放个图

投射层的作用是学习如何从单模态变到多模态

利用**对称式的目标函数**(对比学习的标配）

CLIP 的方式跟之前的对比学习没有本质区别，差异主要是单模态换为了多模态。

两个有趣的细节：
由于数据集非常大:
1. 所以不存在之前同类工作里的 overfiting 的事情，所以很简单。文本和图像编码器都不需要提前训练
2. 使用了线性投射层（不像之前的 SImLR，MoCO 等用了非线性，高10个点）
3. 不需要复杂的数据增强，唯一用的是随机裁剪
4. 由于数据量太大，模型太大，实在不好调参数，所以把 temperature 这个对比学习里非常重要的参数设置为标量，自己学习了

### 2.4 Choosing and Scaling a Model
```
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t) #

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0) # 交叉熵是什么意思？
loss_t = cross_entropy_loss(logits, labels, axis=1) # 
loss = (loss_i + loss_t)/2
```
### 2.5 Training
视觉这里用了8个：5 ResNets，3个 VIT。

训练好之后，又在更大尺寸上 finetune（ViT-L/14@336px)，方法来自 FixRes 这篇论文

## 3 实验
### 3.1 Zero-Shot Transfer

动机：之前的工作(MOCO, SimCLR, DINO) 都是在学习一种表征能力(representation learning)，即使你学到了很好的特征，但你一旦放到下游任务就又涉及到各种下游数据去微调，所以又涉及复杂的事情：比如下游任务可能不好收集数据，数据分布飘逸(distribution shift)不一样。能否训练一个模型，到下游就不需要微调了呢？用文本作为引导