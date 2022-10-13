
Alpa 使用 Python 和 C++来实现，Jax 是 frontend，XLA 是 backend

## Compiler passes：
使用 Jax和 XLA 的IR（即 Jaxpr 和 HLO）。

## Distributed Runtime
Ray actor 来实现 device mesh worker

XLA runtime 来执行计算

## Pytorch Frontend
In general the PyTorch frontend should work since we are using mature PyTorch graph capture technology (torch.fx) and we lower the graph to JAX ops very transparently (done in Python). If you found any bug please feel free to report and we’d love to fix it.
https://alpa-project.slack.com/archives/C0337HQQNAJ/p1658894969365419?thread_ts=1658884081.726559&cid=C0337HQQNAJ
