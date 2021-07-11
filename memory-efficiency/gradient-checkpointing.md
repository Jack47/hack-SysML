1. 在解决的是什么问题？ 优化显存占用，能够训练更大的模型，更大的 batch size
2. 为何成功，标志/准是什么？ 容纳10倍大的模型，只增加20～30% 算力
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？ 内存占用是O(sqrt(n))，额外带来20%～30%的计算开销
5. 有哪些局限性？如何优化？ 如何按照某些原则自动划分 checkpoint ？
6. 这个工作可能有什么深远的影响？


[Visual Gifs to show gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing)

在图里，被选为 checkpoint 的节点，需要在forward 及之后都保持在内存里，而剩余的节点最多只重计算一次。这样重计算的节点数量在 sqrt(n) 级别。对于非 transformer 的网络结构，需要用户手工指定 checkpoints 的位置。

有几种用法来指定 checkpoint:

1. 手工指定：我们在定义 model 时，指定。
2. memory： 使用启发式方法，自动选择一系列节点来 checkpoint。比如把图分成两部分的连接点。
3. speed：最大化运行速度，通过 checkpoint 那些重计算比较慢的节点，比如卷积和矩阵乘


## 几个函数
Tensor.detach() : Returns a new Tensor, detached from the current graph. The result will never require gradient. Returned Tensor shares the same storage with the original one.


## 问题：
1. phony 这个 size 为 1 的占位 tensor 干嘛用的？
2. Checkpointing 里的实现没看到递归找到 Checkpoint，执行一遍 forwawrd 的地方呢？ fairscale 里只是自己实现了部分逻辑： checkpoint ，这个对指定的一个 module 整个进行checkpoint 和 recompute的计算，它并不是最终暴露给用户的接口。看 [PyTorch 的实现](https://github.com/pytorch/pytorch/blob/5c25f8faf3d0d125aa5d642a23b24af2293ade7f/torch/utils/checkpoint.py#L92)比较好，因为完整。
3. 而且要区分是否第二遍forward，第一遍时释放掉，第二遍时不能释放，用完才释放。对，第一遍时，只是让网络执行 forward，而且是 require_grads = False 来执行，所以不会进行动态图构建，所以无法进行 backward 操作，不会自动计算出梯度函数。PyTorch 里是把 forward顺序记录到了一个 function list 里。第二遍 recompute 时，会开启 auto grad，然后 forward，然后 backward。
4. 正常 backward 时，是如何找到上一层的呢？是 pytorch 内部自己实现内的，研究员只需要掉用整个 module 的 backward 即可
5. 在 [fairscale]()/[pytorch]() 的实现里，没找到backward时，先找到最近的 checkpoint 的点，然后 recompute 一遍的地方呢？ 它利用的还是 PyTorch 里实现的 checkpoint_sequential(functions, segments, input, **kwargs)。只不过自己实现了 checkpoint,这个给一个 module 整体进行 checkpoint的函数：保存输入tensor 和参数，forward 时只计算最终 loss，backward 时先 recompute，再backward
6. 如果模型有嵌套，如何处理 ? 只要 checkpoint 的节点划分好了，其他都好说。主要是 checkpoint 划分这里需要考虑
7. 什么情况下适合用？当这段逻辑重计算不复杂，比如非卷积和矩阵乘运算，而激活值和梯度有比较大的情况，比如 transoformer 里参数就很多
8. checkpoint 之间，如何传递 loss 的？ 假设就俩层，那么后一层打开 grad，然后 forward 之后，backward 时就算出了 grad，然后等前一个也打开，backward 时就自动能拿到后一个的 grad 来计算？gg

## PyTorch 实现

### 对外呈现的接口：

1. checkpoint\_sequential(functions, segments, input, **kwargs) // segments 是说当前的 sequential 模型，划分为几段，functions 必须是 nn.Sequential 的

2. `hidden_states = torch.utils.checkpoint.checkpoint(layer_module, hidden_states, attention_mask)` 就是显示指定某一段把开头当作 checkpoint 位置

### 内部辅助函数：

run\_function 作用是把给定的一小段 模型里的层，串出一个顺序执行的顺序来。方便 backward 时执行 forward 过程。因为第一阶段 forward 时没有开启 requires_grad,所以 PyTorch 不会记录动态图。需要我们记录执行流程。

这样它对外呈现的是一个聚合后的一个能执行 forward 的对象。方便在 CheckpointFunction 里把它当作一个 module 来执行一次 forward 函数。理解起来简单

```
def run_function(start, end, functions):
        def forward(input):
```

这个函数是真正实现 checkpoint 一段网络语义下，这段网络的 forward 和 backward 实现

CheckpointFunction(torch.autograd.Function):


## FairScale 的增强

支持在BN的时候不计算 mean  和  variance。但这样结果不就不对了么？

支持把 Tensor offload 到 cpu 内存里
