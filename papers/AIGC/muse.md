1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## [简介](https://muse-model.github.io/)
Muse 是一个 text-to-image 的 Transformer 模型，在比 diffusion 或者 autoregressive 模型更高效的情况下，达到了图片生成性能（质量和差异性）上的 SOTA。Muse 是就一个掩码的模型任务，在离散的token空间上训练的：给定从预训练好的大的语言模型(LLM)上的文本的 embedding，Muse 可以被训练来**预测随机掩码的图片token**。跟 pixel-space 下的diffusion 模型相比，比如 Imagen 和 DALL-E 2， Muse显著地更高效，原因是使用了离散的 tokens，需要更少的迭代次数。与自回归模型，比如 Parti相比，Muse更高效，原因是使用了并行解码。用的预先训练的 LLM 让细粒度的语言理解，转换到高质量(high-fidelity)的图片生成，理解了图形概念(visual concepts)比如目标，他们之间的空间关系，姿势，基数(cardinality)等。我们的9000M（9亿）参数的模型，在 CC3M 上达到了 SOTA，FID 分数是6.06。而 Muse 3B 参数模型在 zero-shot COCO 评估上达到了 7.88 的 FID，CLIP score 是0.32. Muse 也直接可以进行图片编辑应用，无须 fine-tune 或者转换模型：inpainting，outpating，和mask-free 编辑

Muse 的生成速度：515x512 分辨率下，TPUv4上只需要 1.3s。

## 1. 简介
基于文本prompts来调节生成图像的模型，在过去几年里，呈现了质量和灵活性方面的极大飞跃。原因是结合了DL架构创新；训练范式(paradigms) 上的创新，比如在语言和视觉上的 masked modeling；还有新的生成模型比如 diffusion 和 msking-based generation；最后，还有大规模的图片-文本对的训练集。

## 参考资料：

### 去年一年里的变化

Hierarchical text-conditional image generation with clip latents(2022)

Glide: Towards photorealistic image generation and editing with text-guided diffusion models(2021)

Photorealistic text-to-image diffusion models with deep language understanding

### dl 里的架构创新:
Neural discrete representation learning(2017)

Attention is all you need(2017)

### 训练范式创新，比如 masked modeling for both lm and vision tasks
Bert: Pre-training of deep bidirectional transformers for language understanding.(2018)

Exploring the limits of transfer learning with a unified text-to-text transformer: T5(2020)

Masked autoencoders are scalable vision learners(2022)

Maskgit: Masked generative image transformer(2022)

### 新的生成模型
Denoising diffusion probabilistic models(2020)

High-resolution image synthesis with latent diffusion models (2022)

Photorealistic text-to-image diffusion models with deep language understanding(2022)

