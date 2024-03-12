from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP
from transformers.activations import QuickGELUActivation
from thop import profile
from functools import partial
import ipdb
import os
import torch
import time
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from typing import Any, Optional, Tuple, Union


def test_time(test_fn, test_times, warmup_fn, warmup_times, report, report_title):
    if warmup_fn=="same":
        warmup_fn = test_fn
    
    for i in range(warmup_times):
        warmup_fn()

    time_record = []
    for i in range(test_times):
        torch.cuda.synchronize()
        begin = time.time()
        test_fn()
        torch.cuda.synchronize()
        end = time.time()
        time_record.append((end - begin)*1000)

    time_record = np.array(time_record)
    mean = np.mean(time_record)
    std = np.std(time_record)

    if report==True:
        print("\n [ ",report_title," ] \n")
        print("mean: ",mean, " ms")
        print("std: ",std, " ms\n")

    return mean.tolist(), std.tolist()

def test_model(model, inputs):
    return model(inputs.input_ids,inputs.pixel_values, inputs.attention_mask)

class CLIPFlashAttention(torch.nn.Module):
    """
    A wrapper of flashattention
    """
    def __init__(self, clip_attention):
        super().__init__()
        self.clip_attention = clip_attention
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # For simplicity reasons, this version does not consider 'attention_mask' and 'output_attentions'.
        assert attention_mask is None, attention_mask
        assert output_attentions == False
        bsz, tgt_len, embed_dim = hidden_states.size()

        # Input projection
        query_states = self.clip_attention.q_proj(hidden_states).to(torch.float16)
        key_states = self.clip_attention.k_proj(hidden_states).to(torch.float16)
        value_states = self.clip_attention.v_proj(hidden_states).to(torch.float16)

        query_states = query_states.view(bsz, tgt_len, self.clip_attention.num_heads, -1)
        key_states = key_states.view(bsz, tgt_len, self.clip_attention.num_heads, -1)
        value_states = value_states.view(bsz, tgt_len, self.clip_attention.num_heads, -1)

        # Call flash attention
        attn_output = flash_attn_func(q=query_states, k=key_states, v=value_states, 
                        dropout_p=self.clip_attention.dropout,
                        softmax_scale=self.clip_attention.scale,
                        causal=causal_attention_mask is not None).to(torch.float32)

        if attn_output.size() != (bsz , tgt_len, self.clip_attention.num_heads, self.clip_attention.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz , tgt_len, self.clip_attention.num_heads, self.clip_attention.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.reshape(bsz , tgt_len, embed_dim)

        # Output projection
        attn_output = self.clip_attention.out_proj(attn_output)
        
        return attn_output, None


class CLIPFlashAttentionFP16(torch.nn.Module):
    """
    A wrapper of flashattention
    """
    def __init__(self, clip_attention):
        super().__init__()
        if isinstance(clip_attention, CLIPAttention):
            self.clip_attention = clip_attention
        elif isinstance(clip_attention, CLIPFlashAttention):
            self.clip_attention = clip_attention.clip_attention
        self.clip_attention.half()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # For simplicity reasons, this version does not consider 'attention_mask' and 'output_attentions'.
        assert attention_mask is None, attention_mask
        assert output_attentions == False
        bsz, tgt_len, embed_dim = hidden_states.size()

        # Flashattention requires input.dtype is fp16 or bf16
        hidden_states = hidden_states.to(torch.float16)

        # Input projection
        query_states = self.clip_attention.q_proj(hidden_states)
        key_states = self.clip_attention.k_proj(hidden_states)
        value_states = self.clip_attention.v_proj(hidden_states)

        query_states = query_states.view(bsz, tgt_len, self.clip_attention.num_heads, -1)
        key_states = key_states.view(bsz, tgt_len, self.clip_attention.num_heads, -1)
        value_states = value_states.view(bsz, tgt_len, self.clip_attention.num_heads, -1)

        # Call flash attention
        attn_output = flash_attn_func(q=query_states, k=key_states, v=value_states, 
                        dropout_p=self.clip_attention.dropout,
                        softmax_scale=self.clip_attention.scale,
                        causal=causal_attention_mask is not None)

        if attn_output.size() != (bsz , tgt_len, self.clip_attention.num_heads, self.clip_attention.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz , tgt_len, self.clip_attention.num_heads, self.clip_attention.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.reshape(bsz , tgt_len, embed_dim)

        # Output projection
        attn_output = self.clip_attention.out_proj(attn_output)

        # Do I need to turn it back to fp32 here?
        # I concerned that the accuracy of fp16 may not be sufficient when calculating layernorm
        attn_output = attn_output.to(torch.float32)
        
        return attn_output, None



def replace_attention_with_flashattention(model):
    """
    Replace all CLIPAttention in the model with CLIPFlashAttention
    """
    for name, child in model.named_children():
        if isinstance(child, CLIPAttention):
            setattr(model, name, CLIPFlashAttention(child))
        else:
            replace_attention_with_flashattention(child)

def replace_flashattention_with_FP16_version(model):
    """
    Replace all CLIPAttention in the model with CLIPFlashAttention
    """
    for name, child in model.named_children():
        if isinstance(child, CLIPFlashAttention):
            setattr(model, name, CLIPFlashAttentionFP16(child))
        else:
            replace_flashattention_with_FP16_version(child)

def replace_mlp_with_compiled_version(model):
    """
    Replace all CLIPMLP in the model with it's compiled version.
    torch.compile will do the operator fusion, which can reduce the global memory access on GPU.
    """
    torch.set_float32_matmul_precision('high')
    for name, child in model.named_children():
        if isinstance(child, CLIPMLP):
            setattr(model, name, torch.compile(child,mode="max-autotune"))
        else:
            replace_mlp_with_compiled_version(child)

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = CLIPModel.from_pretrained("./models/clip-vit-large-patch14-336").cuda()
    processor = AutoProcessor.from_pretrained("./models/clip-vit-large-patch14-336")
    image = Image.open("000000039769.jpg")
    text = "Two cats are lying on the sofa and sleeping."
    batch_size = 16
    inputs = processor(
        text=[text]*batch_size, images=[image]*batch_size, return_tensors="pt", padding=True
    )
    inputs.input_ids = inputs.input_ids.cuda()
    inputs.pixel_values = inputs.pixel_values.cuda()
    inputs.attention_mask = None

    output1 = test_model(model, inputs).logits_per_image

    test_times = 10
    warmup_times = 4
    test_time(
        test_fn=partial(test_model, model=model, inputs=inputs), 
        test_times=test_times, 
        warmup_fn="same", 
        warmup_times=warmup_times, 
        report=True, 
        report_title="clip-vit-large-patch14-336")

    replace_attention_with_flashattention(model)
    output2 = test_model(model, inputs).logits_per_image
    assert torch.allclose(output1, output2, rtol=1e-02, atol=1e-02)
    del output2
    test_time(
        test_fn=partial(test_model, model=model, inputs=inputs), 
        test_times=test_times, 
        warmup_fn="same", 
        warmup_times=warmup_times, 
        report=True, 
        report_title="clip-vit-large-patch14-336 FlashAttention")

    replace_flashattention_with_FP16_version(model)
    output3 = test_model(model, inputs).logits_per_image
    assert torch.allclose(output1, output3, rtol=1e-02, atol=1e-02)
    del output3
    test_time(
        test_fn=partial(test_model, model=model, inputs=inputs), 
        test_times=test_times, 
        warmup_fn="same", 
        warmup_times=warmup_times, 
        report=True, 
        report_title="clip-vit-large-patch14-336 FlashAttention + FP16")
    
    replace_mlp_with_compiled_version(model)
    output4 = test_model(model, inputs).logits_per_image
    assert torch.allclose(output1, output4, rtol=1e-02, atol=1e-02)
    del output4
    test_time(
        test_fn=partial(test_model, model=model, inputs=inputs), 
        test_times=test_times, 
        warmup_fn="same", 
        warmup_times=warmup_times, 
        report=True, 
        report_title="clip-vit-large-patch14-336 FlashAttention + FP16 + CompiledMLP")
