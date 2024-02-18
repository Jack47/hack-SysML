TLDR:

1. 是 latent diffusion transformer 模型，跟其他的 U-net 视频模型不同
2. 使用 video encoder/decoder：在 latent space 上做 diffuse
3. 在视频的原始尺寸上进行训练，而非 square cropping 数据（需要做物体的居中）
4. 3D consistency 是由于大规模训练而涌现的（不是显示训练出来的
5. Object permanence。即使被其他物体挡住或者离开了帧
6. scaling 大大提升了质量
7. 限制：物理学，物体间交互

# Video generation models as world simulators
我们探索了在视频数据上的大规模生成模型的训练（openai的味道，就是走生成，而非）。我们训练了文本控制的 diffusion models，是在时间变长的视频和图片，分辨率，长宽比(aspect ratio)。才用了 transformer 架构，在视频的 **spacetime patches** 和 **image latent code** 上操作。我们最大的模型 Sora，能够产生一分钟的 high fidelity video。我们的结果说明把 video 生成模型进行 scaling 是一个通往通用目的的物理世界模拟器的有希望的路径之一。

本报告聚焦在：
1. 我们把各类(是不同规格的？)视觉数据变成统一的表征，而且可以大规模训练的生成模型的方法
2. 定量评估 Sora 的能力和限制

## Turning visual data into patches
LLM 范式成功的原因是使用 token 来优雅地统一了文本里的不同模态：代码、数学和不同的自然语言。通用的泛化能力是从大规模数据上学到的。

类似 LLM 里有 text tokens，Sora 有 visual patches。

![](imgs/latentspace-spacetime-patches.png)

从更高纬度而言，我们：
1. 首先把视频变成 patches：压缩到低纬的 latent space：上图里中间那个矩阵
2. 后续把表征解构到 **spacetime patches**: 上图里最右边的队列？

## Video compression network
我们训练了一个网络，可以把视觉数据的纬度缩小。输入是原始的视频，输出是 latent representation ，它在时间和空间上都进行了压缩。不就是类似 VAE 的东西？（看看台大李宏毅关于 diffusion model 的解释）。后续 Sora 就是在这个压缩后的 latent space 上操作的。我们也训练了一个对应的 decoder 模型来把生成的 latents 映射回 pixel space

## Spacetime latent patches
对于上述压缩后的视频，抽取出一个 spacetime patches 的序列，跟 transformer（llm）tokens 类似。这个在图片上也适用，因为图片就是单帧的视频。这个基于 patch 的表征就可以让 Sora 可以在不同分辨率，时长和长宽比（画幅）的视频和图片上训练。跟 LLM 类似？句子、段落、格式都是不同的。

在推理的时候，可以通过排布随机初始化的 patches 到合适的 grid 大小来控制最终生成的视频大小

## Scaling transformers for video generation
Sora 是一个 diffusion model：给定 noisy patches(和类似 text prompt的调控信息），它被训练为预测原始的“干净”的 patches。重要的是，它是 diffusion transformer。[26: ]

Transformer 在大量的领域里证明具有非常有效的 scaling 特性，包括 lm[13, 14], cv[15,16,17,18], image generation[27, 28, 29]

本工作里，发现 diffusion transformer 同样能在 video 模型上也能高效扩展。下面是固定种子和输入后，在训练过程中不同阶段生成视频的样例。样例的质量会随着算力的增加(可能是多训了 epoch，也可能是一个epoch里训的时间越旧？）而明显提高

## 变化的时长、分辨率、画幅
过去的方法是会 resize、crop 和 对齐 video 到标准的大小，比如4秒的 256x256 分辨率。我们发现在原始大小(native size)上训练提供了几种好处：
### Sampling flexibility

### Improved framing and composition

## Language understanding
需要视频和配套的文本标题对。我们使用 DALL E 3 里用到的 重新打标的技术到视频上:

首先训练一个 **高描述性** 的 captioner 模型，然后使用它来产生训练集里所有视频对应的文本标题对。发现高质量的描述性视频标题不仅提高了文本的 fidelity，而且也提高了整体视频的质量

类似 DALL*E 3，我们也使用了 GPT 来把短的用户 prompt 扩展到更长的详细描述的标题上，发给视频模型。这样 Sora 能产生高质量的视频，准确地跟随用户意图

## Prompting with images and videos
支持给定图片或者视频，来基于这些产出视频
### Extending generated videos
可以用来生成无限长度的视频：比如在时间上向前或者向后生成

### Video-to-video editing
我们使用 SDEdit 到 Sora 上。这个技术让 Sora 可以 zero-shot 的形式去变换输入视频的风格和环境
### Connecting videos
可以把两个输入的视频，逐渐结合到一起，产生无缝切换的视频场景

### Image generation capabilities

## Emerging simulation capabilities
发现在大规模数据集上训练后，视频模型展示出一些有趣的涌现能力。让 Sora 可以模拟出一些人类，动物和物理世界的行为。这些属性并没有显示的 3D，物体等地方的规约偏见--他们只是纯粹的规模化之后的产物。

** 3D consistency**：能产生动态相机移动视角。随着相机视角的移动和转动，人类和场景元素会在三维空间上一致地移动

** Long-range coherence and object permanence**  当然也有失败的时候

** Interacting with the world** 可以模拟出真实世界：比如画家画画的过程，一个人吃汉堡的过程

** 模拟电子世界** 比如视频游戏，可以模拟控制 Minecraft 里的任务，这个能力可以通过 “Minecraft” 来触发

## 疑问
1. 视频也是自回归的吗？不需要标注。可能通过看 DiT 能知道个大概？或者看看 GPT4v 的报告
Language understanding 里有写，需要视频和配套的文本标题对。我们使用 DALL E 3 里用到的 重新打标的技术到视频上
2. 视频的长宽比、分辨率、时长等如何控制？


Sora 技术报告，Xie Saining 这么说：

### 架构
Sora 基于 DiT(diffusion transformer)架构，简而言之 DiT = [VAE encoder + ViT + DDPM + VAE decoder]

视觉编码网络：看起来就是一个 VAE，只不过在 raw video data 上训练。Tokenization 可能在得到良好的时间一致性上非常重要。VAE 是一个 ConvNet，所以DiT 也是一个混合网络

DiT 的创新性“不强”，主要精力放在简单和扩展性方面。

* 简单：意味着灵活。ViT 里处理输入数据时很灵活，MAE(masked autoencoder) 里，ViT 可以只处理可见的 patches，忽略 masked ones。类似的，Sora “可以在推理的时候，可以通过排布随机初始化的 patches 到合适的 grid 大小来控制最终生成的视频大小“，而 UNet 并不直接提供这种灵活性。

观察：Sora 可能也使用了 Patch n' Pack (NaViT)，来让 DiT 可以适配到变化的分辨率/时长/画幅

* 扩展性：是 DiT 论文的核心。首先优化后的 DiT 在每个 Flop 上会比 UNet 快。更重要的是 DiT scaling law 不仅能应用在图片，也能放到视频上 -- Sora 复制了在 DiT 里观察到的 visual scaling 行为

观察：根据 "Scaling transformer for video generation"一节里的图，假设 DiT XL/2 的模型就是图里的 basemodel 大小，那可以大概估计出 16倍算力下，就是 3X DiT-XL 的大小，大概是 3B。所以我绝得很快会有新的迭代出来。

最重要的是从 “Emerging simulation capabilities" 这一节。在 Sora 之前，人们并不清楚长时间一致性到底是自己涌现还是需要复杂的主体驱动的生成管线，甚至需要物理模拟器。OpenAI 展示了可以通过端到端训练来达到不是那么完美的模拟能力。但是有两块非常重要的点没被讨论到：

1. 训练数据：来源和如何构造都没提到。可能意味着数据是 Sora 成功的最关键因素。观察：已经有不少游戏引擎的猜测。我也认为电影、纪录片(documentaries)、电影长镜头（long takes)等。质量至关重要。非常好奇 Sora 从哪里拿到的数据（肯定不是 YouTube，对吧？）

2. （Auto-regressive）长视频生成：一个显著突破是它能生成非常长的视频。生成一个2秒的视频和一个一分钟的视频之间是跨跃性的。在 Sora 里，可能是通过允许自回归采样的联合多帧预测里来达到的，其中最大的挑战是如何解决误差累积，在时间域上保持质量和一致性。一个非常长的（双向）的调节用的上下文？或者通过弱化这个问题来 scaling up？这些技术细节非常重要


