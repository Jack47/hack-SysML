1. 在解决的是什么问题？1.描绘文本提示词 2.组合多种名词、形容词和动词
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

two loss functions

## 4. 试验
数据集：

多尺度训练：设置了9个不同的 image scales，会把图片 crop 到离它最近的 bucket。然后根据 bucket 里的图片数量来决定全局的 GPU 资源怎么分配

AdamW 的优化器

## 用到的 SR-GAN 模型
Paper：Real-esrgan: Training real-world blind super-resolution with pure synthetic data
github: https://github.com/xinntao/Real-ESRGAN 效果看着真不错。有一个对不同 upscaler 的介绍：https://www.reddit.com/r/StableDiffusion/comments/y2mrc2/the_definitive_comparison_to_upscalers/

Stable Diffusion X4 Upscaler: https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

提取出的文本 tokens 是被一个 cross-attention 层来放到 U-Net 里去的。

Learning transferable visual models from natural language supervision(2021)(CLIP)

