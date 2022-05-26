Imagen, 是一个文本到图片的扩散模型(diffusion model)，有SOTA的照相写实风格和对自然语言的深度理解。它构建于大规模transformer语言模型对文本的理解能力之上，结合了高保真图片生成的传播模型的优势。
我们关键的发现是通用大规模语言模型(e.g. T5)，只在文本语料库上预训练，能在编码文本合成图片上有奇效：增加语言模型的大小，能在保真度和图片到文本对齐上，相比增大图片传播模型要更有效果。Imagen 达到
新的世界最先进的 COCO数据集上的 FID 分数：7.27，没有在 COCO 上训练，人类评分员发现 Imagen 例子可以和 COCO 数据里 image-text 对齐想匹配。为了进一步深度分析 text-to-image 模型，我们引入了
DrawBench，一个有理解力并且挑战性的text-to-image模型的测试集基准。有了 DrawBench，我们可以和最近包括 VQ-GAN+CLIP，Latent Diffusion Models, DALEE-E 2 进行对比。发现人类评分员更喜欢 Imagen
生成的图片，在样例质量和图片-文本对齐方面。

分为纯语言模型和图像生成模型
## 问题
1. COCO 数据集不是检测吗？还有文本生成图片的测试集？
2. 提到的image-text alignment 是啥东西？
