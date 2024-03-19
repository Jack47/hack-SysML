## 1. 概述

本文对CLIP模型在单卡A100上的前向传播进行加速，综合使用了FlashAttention、混合精度训练、编译的方法，最终达到83%的加速效果。
## 2. 相关配置

模型配置：

- 模型：[CLIP-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
- 图片输入大小：长宽336，通道3，批大小16
- 文本输入大小：序列长度10，批大小16

环境配置：

- GPU：A100
- Python版本：3.9.12
- pytorch版本：2.1.2
- cuda版本：11.8

测试方法：

- 对于每种加速方法，先做4次warmup，然后连续执行10次，记录执行延迟的均值和标准差
- 在python端用time.time()进行计时，计时前使用torch.cuda.synchronize()进行同步。
- 这一版仅考虑了前向传播

## 3. 加速

加速效果总览：

| 方法                              | 延迟（ms）     | 提升（相比baseline） |
| ------------------------------- | ---------- | -------------- |
| baseline                        | 410（±0.3）  | 0%             |
| flash-attn                      | 350（±0.4）  | ↑ 14%          |
| flash-attn + quantize           | 253（±0.65） | ↑ 38%          |
| flash-attn + quantize + compile | 68（±0.5）   | ↑ 83%          |

下面分别说明每种加速方法

### 3.1 使用FlashAttention加速

使用FlashAttention替换掉CLIPAttention中的注意力操作。


- **底层实现**：直接用 https://github.com/Dao-AILab/flash-attention/ 提供的算子实现，调用其封装好的flash_attn_func函数
- **集成**：编写CLIPFlashAttention类，负责完成输入输出的projection、数据类型转换、调用flash_attn_func函数等操作。
- **替换**：自动遍历CLIPModel中的算子，每当遇到CLIPAttention就替换为CLIPFlashAttention

具体实现代码见speedup.py中的CLIPFlashAttention类以及replace_attention_with_flashattention函数。

### 3.2 混合精度

由于flashattention要求输入输出是fp16或bf16，所以在调用flash_attn_func前，需要将Q,K,V转到fp16，并将flash_attn_func的输出转回为fp32。一个很自然的想法是，能否在输入投影前就转到fp16，在输出投影后再转回fp32？
- **好处**：
	- 1）减少数据类型转换的计算量。fp32->fp16的操作减少到1/3。
	- 2）输入、输出的投影操作，也可以在fp16下进行
- **坏处**：有可能影响精度或训练效果

就目前的观察来看，此方法对模型输出影响变化较小。后续可用学习一下pytorch的amp是如何为不同算子赋予不同精度的，或直接把模型扔给amp自动设置精度。

具体代码见CLIPFlashAttentionFP16类和replace_flashattention_with_FP16_version函数

### 3.3 使用torch.compile进行加速

torch.compile可以做一些简单的算子融合。考虑到MLP中的biasadd+quickgelu可以做逐元素的融合，这里尝试直接将CLIPMLP模块喂给torch.compile进行编译。

具体实现见speedup.py的replace_mlp_with_compiled_version函数

>[! note]
>- 性能收益是否仅来自于算子融合所减少的gpu globa memory访问？torch.compile是否有做其他优化？
>- 还有其他算子融合的机会没有尝试，例如可以参考FasterTransformer的做法，将biasadd+resadd+layernorm进行融合
>- 后续还可以尝试，直接把整个模型扔给torch.compile



## 4.执行测试程序

首先需要将[CLIP-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)下载到moodels/CLIP-vit-large-patch14-336中。然后执行speedup.py即可。