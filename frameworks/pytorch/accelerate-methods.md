## 显存相关

缓解显存碎片：

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 加速相关

[TF32](https://github.com/Jack47/hack-SysML/blob/653098577be86b9c8137b3e0f32af204b7c367ab/hardware/GPU/pytorch-cuda.md?plain=1#L70)

torch.backends.cudnn.enabled: it enables cudnn for some operations such as conv layers and RNNs, which can yield a significant speedup. 
