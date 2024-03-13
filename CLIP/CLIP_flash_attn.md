## 一、运行模型

三个模型原生基于不同的开源代码构建，但是能够统一基于 `open_clip` 代码框架来运行，调用的代码如下：

```python
import open_clip
import json
import torch
import numpy as np
from time import time
from calflops import calculate_flops

# CLIP-vit-large-patch14-336
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-L-14-336')

# # laion--CLIP-ViT-H-14-laion2B-s32B-b79K
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
# tokenizer = open_clip.get_tokenizer('ViT-H-14')

# # EVA02_CLIP_E_psz14_s4B
# model, _, preprocess = open_clip.create_model_and_transforms('EVA02-E-14', pretrained='laion2b_s4b_b115k')
# tokenizer = open_clip.get_tokenizer('EVA02-E-14')

# CLIP-vit-large-patch14-336
B = 128
W = 336
H = 336
C = 3
L = 77

# # laion--CLIP-ViT-H-14-laion2B-s32B-b79K
# B = 8
# W = 224
# H = 224
# C = 3
# L = 77

# # EVA02_CLIP_E_psz14_s4B
# B = 1
# W = 224
# H = 224
# C = 3
# L = 77

# 当不设置精度而直接torch.compile()回导致如下warning：
# UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
torch.set_float32_matmul_precision('high') # 充分利用tensor cores来进行矩阵乘法加速
model = torch.compile(model)
model = model.cuda().eval()

image = torch.ones(B, C, H, W).cuda()
text = torch.ones(B, L, dtype=torch.int).cuda()

with torch.no_grad(): # fwd过程中不进行梯度保存，从而节省空间占用，但对于运行速度影响不大
    ts = []
    t0 = time()
    for i in range(14):
        t1 = time()
        output = model(image, text)
        t2 = time()
        ts.append(t2 - t1)
    for _ in range(14): del ts[0]
    print(sum(ts)/len(ts), ts)

flops, macs, params = calculate_flops(model=model, 
                                      input_shape=((B,C,H,W)),
                                      output_as_string=True,
                                      output_precision=4,
                                      print_detailed=False)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# for k, v in model.state_dict().items():
#     print(k, v.shape)
```



## 二、优化attention

在CLIP的模型源代码中，attention模块是直接调用torch.nn.MultiHeadAttention，其底层是基于ATen库用c++实现，暂时未能找到其源码实现。为了检验flash-Attention的效果，我继承并覆写torch.nn.MultiHeadAttention的forward的函数，在其中实现三个版本的forward，分别为：

- `normal_attn` ：不做优化的朴素实现
- `torch_attn`：调用父类实现
- `flash_attn`：使用优化后的flash attention实现，参考https://github.com/Dao-AILab/flash-attention

```python
from flash_attn import flash_attn_func

class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, d_model: int, n_head: int):
        super().__init__(d_model, n_head)
        self.n_head = n_head
        self.d_model = d_model
    
    def normal_attn(self, q, k, v, softmax_scale, attn_mask):
        q = q.transpose(1, 2) # (B, H, L, D//H)
        k = k.transpose(1, 2) # (B, H, L, D//H)
        v = v.transpose(1, 2) # (B, H, L, D//H)
        out = torch.softmax(q @ k.transpose(-1, -2) * softmax_scale + (attn_mask if attn_mask!=None else 0), dim=-1) @ v
        out = out.permute(2, 0, 1, 3)

        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
        return out
    
    def flash_attn(self, q, k, v, softmax_scale, attn_mask):
        out = flash_attn_func(q.half(), k.half(), v.half(), softmax_scale, causal=(attn_mask != None))
        out = out.transpose(0, 1)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = F.linear(out, self.out_proj.weight.half(), self.out_proj.bias.half())
        return out
      
    def torch_attn(self, q, k, v, need_weights=True, attn_mask=None):
        return super().forward(q, k, v, need_weights=need_weights, attn_mask=attn_mask)
    
    def forward(self, q, k, v, need_weights=True, attn_mask=None):
        # q, k, v: (L, B, D)
        # attn_mask: (L, L)
        
        # self.torch_attn(q, k, v, need_weights, attn_mask)
        
        softmax_scale = (self.d_model//self.n_head) ** -0.5  
        qkv = F.linear(q.transpose(0, 1), self.in_proj_weight, self.in_proj_bias) # (B, L, 3*D)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.n_head, -1) # (B, L, 3, H, D//H)
        q, k, v = qkv[:,:,0,:,:], qkv[:,:,1,:,:], qkv[:,:,2,:,:] # (B, L, H, D//H)
        
        out = self.normal_attn(q, k, v, softmax_scale, attn_mask)
        # out = self.flash_attn(q, k, v, softmax_scale, attn_mask)
    
        return out
```

## 三、测试优化效果

运行10次取平均

### 1) Batch_size=1

| CLIP-vit-large-patch14-336    | RTX3090用时              | A100用时                 |
| ----------------------------- | ------------------------ | ------------------------ |
| normal_attn                   | 0.03447s                 | 0.04180s                 |
| torch_attn                    | 0.03305s                 | 0.03750s                 |
| flash_attn                    | 0.02694s (加速比: x1.28) | 0.03067s (加速比: x1.36) |
| flash_attn + tensor cores加速 | 0.02380 (加速比: x1.45)  |                          |

| laion--CLIP-ViT-H-14-laion2B-s32B-b79K | RTX3090用时              | A100用时                 |
| -------------------------------------- | ------------------------ | ------------------------ |
| normal_attn                            | 0.03702s                 | 0.04575s                 |
| torch_attn                             | 0.03714s                 | 0.04739s                 |
| flash_attn                             | 0.03404s (加速比: x1.09) | 0.04340s (加速比: x1.05) |

| EVA02_CLIP_E_psz14_s4B | RTX3090用时 | A100用时 |
| ---------------------- | ----------- | -------- |
| normal_attn            | 显存不足    | 加载报错 |
| torch_attn             | 显存不足    | 加载报错 |
| flash_attn             | 显存不足    | 加载报错 |



### 2) Batch_size=4

| laion--CLIP-ViT-H-14-laion2B-s32B-b79K | A100用时                |
| -------------------------------------- | ----------------------- |
| normal_attn                            | 0.1260s                 |
| torch_attn                             | 0.1222s                 |
| flash_attn                             | 0.1081s (加速比: x1.17) |



### 3) Batch_size=16

| laion--CLIP-ViT-H-14-laion2B-s32B-b79K | A100用时                |
| -------------------------------------- | ----------------------- |
| normal_attn                            | 0.4394s                 |
| torch_attn                             | 0.4435s                 |
| flash_attn                             | 0.3904s (加速比: x1.13) |



### 4) Batch_size=128

| CLIP-vit-large-patch14-336    | RTX3090用时                  |
| ----------------------------- | ---------------------------- |
| normal_attn                   | 3.73205s                     |
| torch_attn                    | 3.34647s                     |
| flash_attn                    | 2.84249s (加速比: x1.31)     |
| flash_attn + tensor cores加速 | 1.58557s (加速比: **x2.35**) |

## 四、思考

目前只在python层面进行了优化，只对MultiHeadAttention中的第二步QKV运算用flash-attention进行了优化，未来可以考虑将QKV运算与in_proj和out_proj融合在一起进行优化成一个单独的算子，这样子可以从cuda层面在底层进行更彻底的优化，因为其实in_proj和out_proj两部分也造成了不小的运算开销。