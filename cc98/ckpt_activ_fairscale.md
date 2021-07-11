# Checkpoint Activation in Pytorch & FairScale

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9ZamhtYmJrZFY2c0lWbmVHaWN3dm1aemZoSDV1Wm1XaWNVTDVBNlB5WjllNW91eGQwb2VFYjExN0hHRW1rR2Y3UjdrbmtLWmlheUVZTFVNOER2dGowY3VPZy82NDA?x-oss-process=image/format,png)

GPT-3, Megatron-LM, Turing-NLG等模型利用巨量参数成为SOTA. 巨型模型的训练依赖于并行和存储优化方法. 可将常用并行优化方法分为数据并行, Tensor模型并行, Pipeline模型并行三类; 常用存储优化方法有: checkpoint activation和cpu_offload两类. Activation partitioning, contiguous checkpointing和cpu checkpointing等存储优化方法也正在发展. 

![](https://github.com/Oldify/my_mdpic/blob/0ed34f9b54f1331d095b495f7a25cb5805150d9a/image-20210711194817143.png)

在三类 (上图华为论文重分为5个维度) 并行架构中, pipeline并行 (上图第e类) 是将整个网络分段 (stage), 不同段在不同的设备上, 前后阶段接力并行. 这一方法既解决了超大模型无法在单设备上装下的难题, 又解决了机器之间的通信开销的问题, 每个stage和下一stage间仅有相邻的某个tensor数据需要传输, 每台机器的数据传输量跟总的网络大小、机器总数、并行规模无关. Pipeline并行依赖于两个重要特性: 梯度累加 (Gradient Accumulation) 和 亚线性内存优化 ([Sublinear Memory Cost](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1604.06174); 陈天奇, 2016).

Checkpoint activation (以下简称重计算) 是上述亚线性内存优化的实现, 在forward时run in torch.no_grad, 丢弃中间激活值, 在backward时依靠存储的内容进行重计算并计算梯度. Megatron和FairScale在实现重计算时, 选择继承torch.autograd, 存储function params; 也有框架采用存储Fake Sub-Graph的思路. 

## 1. Pytorch中的ckpt重计算

在[Pytorch checkpoint activation代码](https://github.com/pytorch/pytorch/blob/666dff381d887eccfe6d1b0ee22cff65e79230d2/torch/utils/checkpoint.py#L260)中已有重计算方法, 更改了forward保存的内容, 保存了function参数, 并在backward时调用. 可以通过以下方法使用:

```python
model = nn.Sequential(...)
input_var = checkpoint_sequential(model, chunks, input_var)
```

也可以在继承nn.Module的自定义模型中添加attr使用:

```python
self.use_checkpoint = use_checkpoint
```

## 2. FairScale中的ckpt重计算

FairScale框架的重计算有三个参数:

- **module** (*nn.Module*) – The module to be wrapped
- **offload_to_cpu** ([*bool*](https://docs.python.org/3.6/library/functions.html#bool)) – Whether to offload activations to CPU.
- **maintain_forward_counter** ([*bool*](https://docs.python.org/3.6/library/functions.html#bool)) – If True, maintain a forward counter per inner module. The counter will first increases in forward calls of outer forward pass and then decreases in the forward calls of outer backward pass. It is used by FullyShardedDataParallel.

使用方法:

```python
checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))
```

FairScale和Pytorch的版本区别在于:

- wraps an nn.Module, so that all subsequent calls will use checkpointing
- handles keyword arguments in the forward
- handles non-Tensor outputs from the forward
- supports offloading activations to CPU

此四点区别通过定义checkpoint_wrapper实现. 在wrapper中, 根据重计算数据来源区分了inner activation (来源于模型inputs和保存的param) 和outer activation (checkpointed modules, 从cpu加载). FairScale版本新定义了_checkpointed_forward, 通过添加dummy tensor with grad克服了原版本只能接受tensor传递的局限, 同时通过weakref避免了内存泄漏.

## 3. 在Transformer中的应用

![Trans](https://github.com/Oldify/my_mdpic/blob/0ed34f9b54f1331d095b495f7a25cb5805150d9a/image-20210711195623058.png)

图中的Q, K, V的参数在实际中 (按列) 被分割成了8份, 输入tensor (按行) 被分割成了16份, 输出tensor因此被分割成了8 × 16 = 128份. 重计算配置在每层内, 引入的多余计算量不超过一层的forward计算量.



## Reference

[Fit More and Train Faster With ZeRO via DeepSpeed and FairScale (huggingface.co)](https://huggingface.co/blog/zero-deepspeed-fairscale)

[Trainer — transformers 4.7.0 documentation (huggingface.co)](https://huggingface.co/transformers/master/main_classes/trainer.html#trainer-integrations)

[Model Parallelism and Big Models · Issue #8771 · huggingface/transformers (github.com)](https://github.com/huggingface/transformers/issues/8771#issuecomment-758418429)

[cybertronai/gradient-checkpointing: Make huge neural nets fit in memory (github.com)](https://github.com/cybertronai/gradient-checkpointing)

[Model Parallelism and Big Models · Issue #8771 · huggingface/transformers (github.com)](https://github.com/huggingface/transformers/issues/8771)

[Activation Checkpoint | FairScale 0.3.7 documentation](https://fairscale.readthedocs.io/en/latest/api/nn/checkpoint/checkpoint_activations.html)

[torch.utils.checkpoint · Pytorch 中文文档 (apachecn.org)](https://pytorch.apachecn.org/docs/1.0/checkpoint.html)

[GPT-3模型为何难以复现？这也许是分布式AI框架的最优设计_OneFlow_Official的博客-CSDN博客](https://blog.csdn.net/OneFlow_Official/article/details/116781168)

[[1604.06174\] Training Deep Nets with Sublinear Memory Cost (arxiv.org)](https://arxiv.org/abs/1604.06174)

[PCL-Platform.Intelligence/PanGu-Alpha: 2000亿开源中文预训练语言模型「鹏程·盘古α」 - PANGU-α.pdf at master - PanGu-Alpha - OpenI](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/PANGU-α.pdf)

