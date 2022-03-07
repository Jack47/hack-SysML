
2021.5.21: 正在探索。希望把编译器优化技术带到pytorch程序更广泛应用里，以便通过 torchscript 来方便地编译，提供更方便的自助服务路径。受
pytorch/xla 启发，它用 lazy tensor 来target 到 xla。但 Alex 团队在 prototyping 方法来泛化使用，看是否可以成为 pytorch 核心的一部分

It provides benefits of fusion or compiler optimization with minimal or no code changes, and a natural 'eager-like' pytorch programming experience.

Torch-XLA provides proof of concept and a fully functional stack for XLA devices, including a lightweight, **purely funtional IR** used for fast DAG construction and hashing, and a **lowering step** for 
conversion to native XLA HLO where optimization and codgen occurs. 

We propose to generalize this **functional trace IR**, make lazy tracing and a backend plugin API a part of Pytorch core

看起来也是增加了一种新的 dispatch key：device 是 lazy



[Lazy Tensor Core Exmaple](https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/example.py#L9-L10) :可以跑跑看？

[API GUIDE, Deep Dive](https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/API_GUIDE.md)

[Lazy Tensor Prototyping](https://github.com/pytorch/pytorch/tree/lazy_tensor_staging) : 开发很活跃。[PyTorch Lazy Tensor Core README](https://github.com/pytorch/pytorch/tree/lazy_tensor_staging/lazy_tensor_core)

[RFC-0012: Functional lazy traces from XLA to PyTorch](https://github.com/pytorch/rfcs/pull/18/files?short_path=1a43f7f#diff-1a43f7ff3f2b5a085ea0067ac6ee48fc1f10d13d0d48bddc0ef328563dd29e3b)

## TorchDynamo (experimental project)
Python-level JIT compilers designed to make unmodified PyTorch programs faster. 

类似工作是 Lazy Tensor Core。两个都想动态捕捉没有修改的 PyTorch 程序。但方法不同：

1. LT : 在 dispatcher level，通过把展开推迟来捕获 op，构建图。
2. TD 在Python frame 级别，跟 JIT 编译器一样。TD只会把frame转换一次，但 LT 每次都捕捉


[貌似还有更高粒度的 TorchDynamo: An Experiment in Dynamic Python Bytecode Transformation](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361)

Next Steps for PyTorch Compilers: 提供 eager mode。Lazy Tensors 是一种方法，但是它在 dispatcher 之下，所以无法移除 Python 的开销和上层 PyTorch Stack。所以对于小，overhead bound 模型


[Next Steps for PyTorch Compilers](https://dev-discuss.pytorch.org/t/next-steps-for-pytorch-compilers/309) : Edge Devices, Compilers in Eager Mode, Next Generation Accelerations, Exploratory Research
