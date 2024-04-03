# 结果

### 区分两个概念

FLOPS： 全大写，指每秒浮点运算次数，可以理解为计算的速度。是衡量硬件性能的一个指标；

FLOPs： s小写，指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。


### 根据config信息估算结果

| 模型规格                                                     | 模型参数量 | TFLOPs |
| ------------------------------------------------------------ | ---------- | ------ |
| [CLIP-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 400M       | 0.51412985856  |
| [laion--CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | 2B         | 0.502825582592  |
| [EVA02_CLIP_E_psz14_s4B](https://huggingface.co/QuanSun/EVA-CLIP) | 4B       | 3.70456649728 |

### thop/torchstat结果

[thop](https://github.com/Lyken17/pytorch-OpCounter)和[torchstat](https://github.com/Swall0w/torchstat)都是基于 PyTorch 的轻量级神经网络分析器，可以用于计算模型的乘加复杂度（MACs）和浮点运算次数（FLOPs），通过调用二者，得到下面的结果，与上述表格结果不一致，但是结果成比例（猜测是设置的输入参数有区别）。

| 模型规格                                                     | 模型参数量 | TFLOPs         |
| ------------------------------------------------------------ | ---------- | -------------- |
| [CLIP-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 400M       | 0.116512260096 |
| [laion--CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | 2B         | 0.10798628864  |
| [EVA02_CLIP_E_psz14_s4B](https://huggingface.co/QuanSun/EVA-CLIP) | 4B         | -              |

**EVA02_CLIP_E_psz14_s4B**加载失败了，可能是模型文件太大或者网络原因。

### fvcore结果

[fvcore](https://github.com/facebookresearch/fvcore)是一个用于PyTorch模型的工具包，包含许多用于模型训练和部署的实用工具，可以提供**操作级**和**模块级** FLOPs 计数。

| 模型规格                                                     | 模型参数量 | TFLOPs         |
| ------------------------------------------------------------ | ---------- | -------------- |
| [CLIP-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 400M       | 0.197764840192 |
| [laion--CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | 2B         | 0.19096870656  |
| [EVA02_CLIP_E_psz14_s4B](https://huggingface.co/QuanSun/EVA-CLIP) | 4B         | 1.156025526016 |

# 问题

### Q0 CLIP 模型的参数量和 CLIP 模型对应的 TFLOPS 之间是什么关系？

在input_size固定的情况下，模型对应的TFLOPs与参数量成正比。

### Q1 TFlops 有哪些计算方法？你使用了哪几种

根据网上搜集的资料，目前计算模型TFLOPs主要可以分为以下几类

#### 1. 模块级（per-module）计数

目前有多种工具（如[pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter), [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch), [mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py), [pytorch_model_summary](https://github.com/ceykmc/pytorch_model_summary)等）可以通过利用PyTorch的模块前向钩子（module forward hooks）来计算每个模块的FLOPs。

可能存在的局限性：

1. **准确性问题**：这些工具的准确性依赖于每个自定义模块都必须实现相应的FLOP计数器。对用户而言，这意味着需要额外付出很多努力，因为他们必须为每个自定义模块编写FLOP计数代码
2. **复杂模块**：如果要确定一个复杂模块的FLOPs计数器，就需要手动检查其前向代码，了解调用了哪些基本操作（raw ops）。此外，如果对模块的前向逻辑进行重构（例如，用子模块替换原始操作），可能还需要更改其FLOP计数器
3. **控制流**：当模块内部包含控制流时，计数代码需要复制一些模块的前向逻辑，这也增加了复杂性和实施难度

#### 2. 操作符级（per-operator）计数

相对于模块级计数，操作符级计数看起来更为合适。因为与用户通常创建的大量自定义模块不同，自定义操作符较少见。同时，操作符通常不包含需要在其FLOP计数器中复制的控制逻辑。因此，为了获得准确的结果，基于操作符级别进行FLOP计数更为理想。

#### 3. 硬件指令级计数

计算实际硬件上的浮点运算次数，`perf stat`能够收集命令的实际指令计数。

我利用三个模型的配置文件中的超参数，实现了一个估算的Python脚本，总体的思路与模块级计数类似。

### Q2 是否能找到这个模型在不同的硬件，比如 A100、H800 等上面的 sota 的 MFU？

没有找到针对CLIP模型的，但是有针对LLM的[MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/pdf/2402.15627.pdf)

### Q3 上面对应的 sota 指标是如何达到的？

1. **算法与系统共同设计**：MegaScale通过全栈方法优化算法和系统组件，这包括模型结构和优化器设计、计算和通信的重叠、操作优化、数据处理流程，以及网络性能调整。这种综合考虑算法和系统层面的方法有助于提高整体训练效率。
2. **通信与计算重叠**：为了减少迭代时间，MegaScale系统精心设计了各种技术来隐藏所有非关键路径操作的开销。特别是，在数据并行性、管道并行性和张量/序列并行性中，MegaScale采用了特定的技术来最大限度地减少通信延迟，并将其与计算过程重叠。
3. **高效的操作优化**：MegaScale对多个操作进行了优化，包括采用更有效的FlashAttention-2实现注意力机制，以及对LayerNorm和GeLU的融合，以减少内核启动的开销并优化内存访问模式
4. **数据管道优化**：MegaScale优化了数据预处理和加载过程，以减少GPU的空闲时间，并实现了异步数据预处理和去冗余的数据加载器设计。
5. **集体通信群组初始化**：对于分布式训练，MegaScale优化了NVIDIA Collective Communications Library（NCCL）通信组的初始化过程，以减少启动时的开销。
6. **模型和优化器的修改**：包括采用并行Transformer块、滑动窗口注意力机制（SWA）和LAMB优化器，这些技术有助于提高模型的训练效率，同时保持精度。

### Q4 我们使用的 clip 会切换调整，如何实现一个 python 函数快速计算当前给定的 clip 规格，返回 TFLOPs ？

```python
import json

def calculate_tflops(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    if "text_config" in config and "vision_config" in config:
        text_config = config["text_config"]
        vision_config = config["vision_config"]
    elif "text_cfg" in config and "vision_cfg" in config:
        text_config = config["text_cfg"]
        vision_config = config["vision_cfg"]

    hidden_size_text = text_config.get("hidden_size", 768) if "hidden_size" in text_config else text_config.get("width", 768)
    num_layers_text = text_config.get("num_hidden_layers", 12) if "num_hidden_layers" in text_config else text_config.get("num_layers", 12)
    # max_position_embeddings = text_config.get("max_position_embeddings", 77)
    max_position_embeddings = text_config.get("context_length", 77) if "context_length" in text_config else text_config.get("max_position_embeddings", 77)

    # 提取图像编码器配置
    hidden_size_image = vision_config.get("hidden_size", 1024) if "hidden_size" in vision_config else vision_config.get("width", 1024)
    num_layers_image = vision_config.get("num_hidden_layers", 24) if "num_hidden_layers" in vision_config else vision_config.get("layers", 24)
    image_size = vision_config.get("image_size", 336)
    patch_size = vision_config.get("patch_size", 14)
    mlp_ratio = vision_config.get("mlp_ratio", 4)

    # 计算序列长度
    seq_length_image = (image_size / patch_size) ** 2

    # 计算前馈网络中间层大小
    intermediate_size_text = hidden_size_text * mlp_ratio
    intermediate_size_image = hidden_size_image * mlp_ratio

    # 计算每层的运算量
    text_per_layer = (4 * hidden_size_text * max_position_embeddings ** 2 +
                      8 * hidden_size_text * intermediate_size_text * max_position_embeddings)
    image_per_layer = (4 * hidden_size_image * seq_length_image ** 2 +
                       8 * hidden_size_image * intermediate_size_image * seq_length_image)

    # 总运算量（乘以层数）
    total_ops = (text_per_layer * num_layers_text) + (image_per_layer * num_layers_image)

    # 转换为TFLOPs
    tflops = total_ops / 1e12
    return tflops

config_path = 'config_1.json'
tflops = calculate_tflops(config_path)
print(f"Calculated TFLOPs: {tflops}")

config_path = 'config_2.json'
tflops = calculate_tflops(config_path)
print(f"Calculated TFLOPs: {tflops}")

config_path = 'config_3.json'
tflops = calculate_tflops(config_path)
print(f"Calculated TFLOPs: {tflops}")
```

这里实现了一个基于config.json文件中的各种超参数来估算模型TFLOPs的Python脚本，其中用到了CLIP模型的以下参数：

1. **Text Configuration Parameters (文本配置参数)**:
   - `hidden_size` 或 `width`: 文本编码器的隐藏层大小
   - `num_hidden_layers` 或 `num_layers`: 文本编码器的层数
   - `context_length` 或 `max_position_embeddings`: 文本编码器的最大位置嵌入数或上下文长度
2. **Vision Configuration Parameters (视觉配置参数)**:
   - `hidden_size` 或 `width`: 图像编码器的隐藏层大小
   - `num_hidden_layers` 或 `layers`: 图像编码器的层数
   - `image_size`: 图像输入的大小
   - `patch_size`: 将图像分割成的小块（patch）的大小
   - `mlp_ratio`: 多层感知器（MLP）的比率，用于计算前馈网络的中间层大小

考虑到[CLIP-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)、[laion--CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)和[EVA02_CLIP_E_psz14_s4B](https://huggingface.co/QuanSun/EVA-CLIP)的配置信息格式不一样，所以程序中最后一个模型的参数格式做了兼容处理。计算公式参考的是https://arxiv.org/abs/2104.04473

### Q5 有没有遇得到一些困难或者不理解的概念？具体有哪些可以展开说说

不太清楚为什么几种方式算出来的TFLOPs有区别但又成比例



