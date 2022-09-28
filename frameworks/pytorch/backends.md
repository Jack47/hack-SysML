## nvFuser
Internally, nvFuser is a **Halide-like** system, and we think Halide's separation of "algorithms" and "schedules" is the best way to accomplish the above goals. 

它是专门给PyTorch 用的，主要是NV的人实现的，帮助根据PyTorch程序生成 cuda 代码

是支持 dynamic shape 的，相当于把 NV 的cuda 优化经验融入到 PyTorch 里，尤其是 Normalizations 算子上速度很快。支持创新的算子

通过 JIT 里的 IR，会转化为自己的nvFuser IR

## [torchInductor](https://github.com/pytorch/torchdynamo/tree/main/torchinductor)
torchinductor 的代码在torchDynamo里,它是PyTorch的新编译器，能够表示 PyTorch 里所有的算子，能够支持训练以及多种不同的后端。团队主要花时间在支持PyTorch 的多样性上，没有花太多时间在优化上。

## 参考资料：
1. GTC 2022：The Next Generation of GPU Performance in PyTorch with **nvFuser**](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41958/): 主要讲了 fusion 是tvm、xla、halide里最重要的优化手段，作用是增强程序局部性（数据在cache 或者寄存器里）
2. [GTC 2021 **nvFuser** Dynamic Shapes First: Advanced GPU Fusion in PyTorch](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31952/): 主要介绍 dynamic shape 是怎么实现的. 不需要频繁编译，就可以支持 dynamic shape

