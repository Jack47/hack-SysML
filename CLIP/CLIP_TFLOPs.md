# 模型参数及TFLOPs

概念理解：TFLOPs 与 TFLOPS 二者是容易混淆的两个概念，TFLOPs 是针对于计算开销，描述运算过程中需要进行多少次浮点运算；TFLOPS则是针对于硬件，是描述硬件每秒(Per Second)进行浮点运算次数的量。

## 模型规格

| 模型规格                                                     | 模型参数量 | TFLOPs |
| ------------------------------------------------------------ | ---------- | ------ |
| [CLIP-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 400M       | 0.349  |
| [laion--CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | 1B         | 0.324  |
| [EVA02_CLIP_E_psz14_s4B](https://huggingface.co/QuanSun/EVA-CLIP) |            |        |

**各模型超参**：

```json
// CLIP-vit-large-patch14-336
{
    "embed_dim": 768,
    "vision_cfg": {
        "image_size": 336,
        "layers": 24,
        "width": 1024,
        "patch_size": 14
    },
    "text_cfg": {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 768,
        "heads": 12,
        "layers": 12
    }
}

// laion--CLIP-ViT-H-14-laion2B-s32B-b79K
{
    "embed_dim": 1024,
    "vision_cfg": {
        "image_size": 224,
        "layers": 32,
        "width": 1280,
        "head_width": 80,
        "patch_size": 14
    },
    "text_cfg": {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 1024,
        "heads": 16,
        "layers": 24
    }
}

// EVA02_CLIP_E_psz14_s4B
{
    "embed_dim": 1024,
    "vision_cfg": {
        "image_size": 224,
        "timm_model_name": "eva02_enormous_patch14_clip_224",
        "timm_model_pretrained": false,
        "timm_pool": "token",
        "timm_proj": null
    },
    "text_cfg": {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 1024,
        "heads": 16,
        "layers": 24
    },
    "custom_text": true
}
```



## TFLOPs计算

为了节省显存，此处统一取batch_size为B=1，经验证TFLOPs与B呈线性正比例关系。

使用 手工推导 + 调 `calflops` 库检验 的方式进行TFLOPs计算

| 模型规格                               | 总参数量    | image_encoder参数量 | image_encoder中Transformer参数量 | text_encoder参数量 | text_encoder中Transformer参数量 |
| -------------------------------------- | ----------- | ------------------- | -------------------------------- | ------------------ | ------------------------------- |
| CLIP-vit-large-patch14-336             | 427,944,793 | 304,293,888         | 302,309,376<br />(占比99.34%)    | 85,054,464         | 85,054,464<br />(占比100%)      |
| laion--CLIP-ViT-H-14-laion2B-s32B-b79K | 986,109,441 | 632,076,800         | 629,678,080<br />(占比99.62%)    | 302,309,376        | 302,309,376<br />(占比100%)     |
| EVA02_CLIP_E_psz14_s4B                 |             |                     |                                  |                    |                                 |

由此可见Transformer的参数量在模型总参数量中占主导部分，而该部分主要由多头注意力机制MultiHeadAttention实现，假设其输入 X 的形状为 (B, L, D)，其中B为batch_size，L为seq_len，D为隐层维度，用H表示num_heads，MultiHeadAttention主体计算包括3部分：

- **in_proj 映射**：由 D 到 3D 的 Linear 层，用于 self_attention 生成 Q、K、V，该部分的FLOPs为：
  $$
  FLOPs = B·L·3D·2D = 6B·L·D^2
  $$

- **多头QKV计算**：(softmax(Q @ K^T * softmax_scale) @ V)，该部分的FLOPs为：
  $$
  FLOPs = B·L^2·(2D-1) + B·L^2 + B·L·(3L-1) + B·L·D·(2L-1) ≈ 4B·L^2·D
  $$
  

- **out_proj 映射**：输出层由 D 到 D 的 Linear 层，该部分的FLOPs为：
  $$
  FLOPs = B·L·D·2D = 2B·L·D^2
  $$
  这三步总的FLOPs为：
  $$
  FLOPs ≈ 8B·L·D^2 + 4B·L^2·D
  $$
  MultiHeadAttention模块参数量：
  $$
  Params = (3D^2 + 3D) + (D^2 + D) = 4D^2 + 4D
  $$
  将两式做比值，得到：
  $$
  \frac{FLOPS}{Params} ≈ \frac{L(2D+L)}{D+1} = 2L + \frac{L(L-2)}{D+1}
  $$
  

### (1) CLIP-vit-large-patch14-336

image_encoder中的Transformer：L=577，D=1024

text_encoder中的Transformer：L=77，D=768

带入上述公式，计算得到
$$
TFLOPs ≈ 0.349 + 0.013 = 0.362
$$
与调 `calflops` 库所得0.349相近

### (2) laion--CLIP-ViT-H-14-laion2B-s32B-b79K

image_encoder中的Transformer：L=257，D=1280

text_encoder中的Transformer：L=77，D=1024

带入上述公式，计算得到
$$
TFLOPs ≈ 0.324 + 0.046 = 0.370
$$
与调 `calflops` 库所得0.324相近

### (3) EVA02_CLIP_E_psz14_s4B





# 附录

## A.各模型完整参数结构

### (1) CLIP-vit-large-patch14-336

| param_name                                                   | param_size                    |
| ------------------------------------------------------------ | ----------------------------- |
| positional_embedding                                         | torch.Size([77, 768])         |
| text_projection                                              | torch.Size([768, 768])        |
| logit_scale                                                  | torch.Size([])                |
| total_ops                                                    | torch.Size([1])               |
| total_params                                                 | torch.Size([1])               |
| visual.class_embedding                                       | torch.Size([1024])            |
| visual.positional_embedding                                  | torch.Size([577, 1024])       |
| visual.proj                                                  | torch.Size([1024, 768])       |
| visual.total_ops                                             | torch.Size([1])               |
| visual.total_params                                          | torch.Size([1])               |
| visual.conv1.weight                                          | torch.Size([1024, 3, 14, 14]) |
| visual.patch_dropout.total_ops                               | torch.Size([1])               |
| visual.patch_dropout.total_params                            | torch.Size([1])               |
| visual.ln_pre.weight                                         | torch.Size([1024])            |
| visual.ln_pre.bias                                           | torch.Size([1024])            |
| visual.ln_pre.total_ops                                      | torch.Size([1])               |
| visual.ln_pre.total_params                                   | torch.Size([1])               |
| visual.transformer.total_ops                                 | torch.Size([1])               |
| visual.transformer.total_params                              | torch.Size([1])               |
| visual.transformer.resblocks.total_ops                       | torch.Size([1])               |
| visual.transformer.resblocks.total_params                    | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].total_ops                | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].total_params             | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].ln_1.weight              | torch.Size([1024])            |
| visual.transformer.resblocks.[0-23].ln_1.bias                | torch.Size([1024])            |
| visual.transformer.resblocks.[0-23].ln_1.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].ln_1.total_params        | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].attn.in_proj_weight      | torch.Size([3072, 1024])      |
| visual.transformer.resblocks.[0-23].attn.in_proj_bias        | torch.Size([3072])            |
| visual.transformer.resblocks.[0-23].attn.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].attn.total_params        | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].attn.out_proj.weight     | torch.Size([1024, 1024])      |
| visual.transformer.resblocks.[0-23].attn.out_proj.bias       | torch.Size([1024])            |
| visual.transformer.resblocks.[0-23].attn.out_proj.total_ops  | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].attn.out_proj.total_params | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].ls_1.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].ls_1.total_params        | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].ln_2.weight              | torch.Size([1024])            |
| visual.transformer.resblocks.[0-23].ln_2.bias                | torch.Size([1024])            |
| visual.transformer.resblocks.[0-23].ln_2.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].ln_2.total_params        | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].mlp.c_fc.weight          | torch.Size([4096, 1024])      |
| visual.transformer.resblocks.[0-23].mlp.c_fc.bias            | torch.Size([4096])            |
| visual.transformer.resblocks.[0-23].mlp.gelu.total_ops       | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].mlp.gelu.total_params    | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].mlp.c_proj.weight        | torch.Size([1024, 4096])      |
| visual.transformer.resblocks.[0-23].mlp.c_proj.bias          | torch.Size([1024])            |
| visual.transformer.resblocks.[0-23].ls_2.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-23].ls_2.total_params        | torch.Size([1])               |
| visual.ln_post.weight                                        | torch.Size([1024])            |
| visual.ln_post.bias                                          | torch.Size([1024])            |
| visual.ln_post.total_ops                                     | torch.Size([1])               |
| visual.ln_post.total_params                                  | torch.Size([1])               |
| transformer.total_ops                                        | torch.Size([1])               |
| transformer.total_params                                     | torch.Size([1])               |
| transformer.resblocks.total_ops                              | torch.Size([1])               |
| transformer.resblocks.total_params                           | torch.Size([1])               |
| transformer.resblocks.[0-11].total_ops                       | torch.Size([1])               |
| transformer.resblocks.[0-11].total_params                    | torch.Size([1])               |
| transformer.resblocks.[0-11].ln_1.weight                     | torch.Size([768])             |
| transformer.resblocks.[0-11].ln_1.bias                       | torch.Size([768])             |
| transformer.resblocks.[0-11].ln_1.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-11].ln_1.total_params               | torch.Size([1])               |
| transformer.resblocks.[0-11].attn.in_proj_weight             | torch.Size([2304, 768])       |
| transformer.resblocks.[0-11].attn.in_proj_bias               | torch.Size([2304])            |
| transformer.resblocks.[0-11].attn.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-11].attn.total_params               | torch.Size([1])               |
| transformer.resblocks.[0-11].attn.out_proj.weight            | torch.Size([768, 768])        |
| transformer.resblocks.[0-11].attn.out_proj.bias              | torch.Size([768])             |
| transformer.resblocks.[0-11].attn.out_proj.total_ops         | torch.Size([1])               |
| transformer.resblocks.[0-11].attn.out_proj.total_params      | torch.Size([1])               |
| transformer.resblocks.[0-11].ls_1.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-11].ls_1.total_params               | torch.Size([1])               |
| transformer.resblocks.[0-11].ln_2.weight                     | torch.Size([768])             |
| transformer.resblocks.[0-11].ln_2.bias                       | torch.Size([768])             |
| transformer.resblocks.[0-11].ln_2.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-11].ln_2.total_params               | torch.Size([1])               |
| transformer.resblocks.[0-11].mlp.c_fc.weight                 | torch.Size([3072, 768])       |
| transformer.resblocks.[0-11].mlp.c_fc.bias                   | torch.Size([3072])            |
| transformer.resblocks.[0-11].mlp.gelu.total_ops              | torch.Size([1])               |
| transformer.resblocks.[0-11].mlp.gelu.total_params           | torch.Size([1])               |
| transformer.resblocks.[0-11].mlp.c_proj.weight               | torch.Size([768, 3072])       |
| transformer.resblocks.[0-11].mlp.c_proj.bias                 | torch.Size([768])             |
| transformer.resblocks.[0-11].ls_2.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-11].ls_2.total_params               | torch.Size([1])               |
| token_embedding.weight                                       | torch.Size([49408, 768])      |
| token_embedding.total_ops                                    | torch.Size([1])               |
| token_embedding.total_params                                 | torch.Size([1])               |
| ln_final.weight                                              | torch.Size([768])             |
| ln_final.bias                                                | torch.Size([768])             |
| ln_final.total_ops                                           | torch.Size([1])               |
| ln_final.total_params                                        | torch.Size([1])               |

### (2) laion--CLIP-ViT-H-14-laion2B-s32B-b79K

| param_name                                                   | param_size                    |
| ------------------------------------------------------------ | ----------------------------- |
| positional_embedding                                         | torch.Size([77, 1024])        |
| text_projection                                              | torch.Size([1024, 1024])      |
| logit_scale                                                  | torch.Size([])                |
| total_ops                                                    | torch.Size([1])               |
| total_params                                                 | torch.Size([1])               |
| visual.class_embedding                                       | torch.Size([1280])            |
| visual.positional_embedding                                  | torch.Size([257, 1280])       |
| visual.proj                                                  | torch.Size([1280, 1024])      |
| visual.total_ops                                             | torch.Size([1])               |
| visual.total_params                                          | torch.Size([1])               |
| visual.conv1.weight                                          | torch.Size([1280, 3, 14, 14]) |
| visual.patch_dropout.total_ops                               | torch.Size([1])               |
| visual.patch_dropout.total_params                            | torch.Size([1])               |
| visual.ln_pre.weight                                         | torch.Size([1280])            |
| visual.ln_pre.bias                                           | torch.Size([1280])            |
| visual.ln_pre.total_ops                                      | torch.Size([1])               |
| visual.ln_pre.total_params                                   | torch.Size([1])               |
| visual.transformer.total_ops                                 | torch.Size([1])               |
| visual.transformer.total_params                              | torch.Size([1])               |
| visual.transformer.resblocks.total_ops                       | torch.Size([1])               |
| visual.transformer.resblocks.total_params                    | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].total_ops                | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].total_params             | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].ln_1.weight              | torch.Size([1280])            |
| visual.transformer.resblocks.[0-31].ln_1.bias                | torch.Size([1280])            |
| visual.transformer.resblocks.[0-31].ln_1.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].ln_1.total_params        | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].attn.in_proj_weight      | torch.Size([3840, 1280])      |
| visual.transformer.resblocks.[0-31].attn.in_proj_bias        | torch.Size([3840])            |
| visual.transformer.resblocks.[0-31].attn.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].attn.total_params        | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].attn.out_proj.weight     | torch.Size([1280, 1280])      |
| visual.transformer.resblocks.[0-31].attn.out_proj.bias       | torch.Size([1280])            |
| visual.transformer.resblocks.[0-31].attn.out_proj.total_ops  | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].attn.out_proj.total_params | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].ls_1.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].ls_1.total_params        | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].ln_2.weight              | torch.Size([1280])            |
| visual.transformer.resblocks.[0-31].ln_2.bias                | torch.Size([1280])            |
| visual.transformer.resblocks.[0-31].ln_2.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].ln_2.total_params        | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].mlp.c_fc.weight          | torch.Size([5120, 1280])      |
| visual.transformer.resblocks.[0-31].mlp.c_fc.bias            | torch.Size([5120])            |
| visual.transformer.resblocks.[0-31].mlp.gelu.total_ops       | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].mlp.gelu.total_params    | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].mlp.c_proj.weight        | torch.Size([1280, 5120])      |
| visual.transformer.resblocks.[0-31].mlp.c_proj.bias          | torch.Size([1280])            |
| visual.transformer.resblocks.[0-31].ls_2.total_ops           | torch.Size([1])               |
| visual.transformer.resblocks.[0-31].ls_2.total_params        | torch.Size([1])               |
| visual.ln_post.weight                                        | torch.Size([1280])            |
| visual.ln_post.bias                                          | torch.Size([1280])            |
| visual.ln_post.total_ops                                     | torch.Size([1])               |
| visual.ln_post.total_params                                  | torch.Size([1])               |
| transformer.total_ops                                        | torch.Size([1])               |
| transformer.total_params                                     | torch.Size([1])               |
| transformer.resblocks.total_ops                              | torch.Size([1])               |
| transformer.resblocks.total_params                           | torch.Size([1])               |
| transformer.resblocks.[0-23].total_ops                       | torch.Size([1])               |
| transformer.resblocks.[0-23].total_params                    | torch.Size([1])               |
| transformer.resblocks.[0-23].ln_1.weight                     | torch.Size([1024])            |
| transformer.resblocks.[0-23].ln_1.bias                       | torch.Size([1024])            |
| transformer.resblocks.[0-23].ln_1.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-23].ln_1.total_params               | torch.Size([1])               |
| transformer.resblocks.[0-23].attn.in_proj_weight             | torch.Size([3072, 1024])      |
| transformer.resblocks.[0-23].attn.in_proj_bias               | torch.Size([3072])            |
| transformer.resblocks.[0-23].attn.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-23].attn.total_params               | torch.Size([1])               |
| transformer.resblocks.[0-23].attn.out_proj.weight            | torch.Size([1024, 1024])      |
| transformer.resblocks.[0-23].attn.out_proj.bias              | torch.Size([1024])            |
| transformer.resblocks.[0-23].attn.out_proj.total_ops         | torch.Size([1])               |
| transformer.resblocks.[0-23].attn.out_proj.total_params      | torch.Size([1])               |
| transformer.resblocks.[0-23].ls_1.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-23].ls_1.total_params               | torch.Size([1])               |
| transformer.resblocks.[0-23].ln_2.weight                     | torch.Size([1024])            |
| transformer.resblocks.[0-23].ln_2.bias                       | torch.Size([1024])            |
| transformer.resblocks.[0-23].ln_2.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-23].ln_2.total_params               | torch.Size([1])               |
| transformer.resblocks.[0-23].mlp.c_fc.weight                 | torch.Size([4096, 1024])      |
| transformer.resblocks.[0-23].mlp.c_fc.bias                   | torch.Size([4096])            |
| transformer.resblocks.[0-23].mlp.gelu.total_ops              | torch.Size([1])               |
| transformer.resblocks.[0-23].mlp.gelu.total_params           | torch.Size([1])               |
| transformer.resblocks.[0-23].mlp.c_proj.weight               | torch.Size([1024, 4096])      |
| transformer.resblocks.[0-23].mlp.c_proj.bias                 | torch.Size([1024])            |
| transformer.resblocks.[0-23].ls_2.total_ops                  | torch.Size([1])               |
| transformer.resblocks.[0-23].ls_2.total_params               | torch.Size([1])               |
| token_embedding.weight                                       | torch.Size([49408, 1024])     |
| token_embedding.total_ops                                    | torch.Size([1])               |
| token_embedding.total_params                                 | torch.Size([1])               |
| ln_final.weight                                              | torch.Size([1024])            |
| ln_final.bias                                                | torch.Size([1024])            |
| ln_final.total_ops                                           | torch.Size([1])               |
| ln_final.total_params                                        | torch.Size([1])               |

### (3) EVA02_CLIP_E_psz14_s4B

