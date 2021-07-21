## TODO
1. 看如何用几行代码实现的 tensor 级别并行，而且是在 g 和 f 那里要all-reduce 一下
2. pretrain_gpt.py
3. timers 里记录的数据从哪里能获得?
4. 从上到下看一下。最近都是在下面看的，比较碎，而且和上面没有连接起来。比如 tensor model parallel 如何实现？
5. 看一下 pipeline 的三个版本实现

## model/transformer.py
ParallelTransformerLayer : LayerNorm -> ParallelAttention -> LayerNorm => ParallelMLP

```
ParallelAttention:
    query_key_value = mpu.ColumnParallelLinear()
    attention_dropout = torch.nn.Dropout(args.attention_dropout)
    dense = mpu.RowParallelLinear()
```

## P2P communication
`send_forward_recv_backward` 
1. 谁调用它？它不涉及并发和重叠吧？就是同步的。
2. 怎么知道应该发给谁？有几个 group：pipeline model parallel group, tensor model parallel group, dp group


### 通信涉及的 groups 和 ranks： p2p_communication.py, mpu/initialize.py
假设16张卡，想利用为两路 tensor 模型并行，共四个stage 的模型并行，两路数据并行，那么：
tensor model parallel size = 2
pipeline model parallel size = 4
data parallel size = 16 / (tensor model parallel size * pipeliine model parallel size) = 2
公式：world size = tensor model parallel size * pipeline model parallel size * data parallel size


groups num：
pipeline model parallel groups: 16 / pipeline model parallel size = 4 。每一路需要在一个 group 里，前后会互相需要激活值或梯度值
tensor model parallel groups: 16 / tensor model parallel size = 8。每一路内部需要同步结构
dp groups: 16 /2 =  8  路
公式： num of groups = world size / x parallel size，属于一个 group 里的人需要互相通信

在构建 groups 时  api 如下：`group = torch.distributed.new_group(ranks)`。所以后续用的group 都是 一个通信具柄，自己所在组。

## 调度? schedules.py
forward step 里，自己这里完了后，把输出结果发给下游
```
p2p_communication.send_forward()
    forward_backward_pipelining_without_interleaving
```
    
## pretrain_gpt.py

## mpu(model parallel{utility, initialize, mapping, layers, random} ) 里的东东
tensor model parallel + pipeline model parallel 的例子在：
*\_distributed\_with\_mp.sh 里

pipeline model parallel 参数：
pipeline-model-parallel-size 2

split 949,50,1 // 数据划分为三个：training/validation/test 集合
global-batch-size 16
micro batch size 4

tensor model parallel 参数：


tensor model parallel 时，先把输入按照 world size 划分一下，这样每个 rank 自己算，forward 之后再看情况合并（可能在当前算子里，也可能在之后必须要合并的地方）

split <-> all gather :  切分和合并，维度是会变化的

all reduce: 维度不变，把大家用 sum op 搞成一致的内容

`split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False)` : 就是调用 `torch.split(tensor, last_dim_size, dim=last_ddim)`

`split_tensor_into_1d_equal_chunks` : pipeline model parallel 里发送给上下游时，为了充分利用两台主机间的多张 IB 卡，可以使用 split/gather 方式并发发送 tensor。因为同group里大家的 tensor 都应该相同，所以就单纯split 一下，然后发送自己负责的即可

## Tensor Model Parallel
### ColumnParallelLinear 
1. `__init__` 时，初始化参数只给自己负责的部分做即可：`output_size_per_partition = output_size / tensor_model_parallel_world_size`
2. forward 的实现很好玩：

```
input_parallel = copy_to_tensor_model_parallel_region(input_)
```
上述看起来是一个函数，要拷贝。实际背后是 Function：forward 时 split(scatter)，backward 时 gather

forward 时 split，当 backward 时先 gather

## DDP

AllgatherFromDataParallelRegion 也是实现为一个 Function：forward 时是all gather，找好自己的位置，然后 all gather。backward 时就只需要把梯度split，然后用自己的那部分

## micro batch size, global batch size 参数如何起作用的？
参考 training.py:

`micro_batch_size`

```
global_batch_size = dist.get_world_size() * micro_batch_size`
num_micro_batches = global_batch_size // micro_batch_times_data_parallel # 看起来就是i全局来看，需要多少次 micro_batch
所以要一个 batch 里消耗的 sample 数量 = data_parallel_world_size() * micro_batch_size * num_microbatches() # 也是等于 global_batch_size 吧，只不过这个也是算出的
```
batch 是直接在代码层面当作数据里的一个维度来处理？这样上层看到的就是batch size 个数据放到一个 tensor 里进行后续处理。而 micro batch 是为了在大 batch 情况下节省内存，即较小规模地进行 forward/backward，等凑够 batch size 之后，再做多个副本之间的同步？

## dataloader 用到的 sampler
dataloader 没啥特殊的
```
data/data_samplers.py
    MegatronPretrainingSampler // 给 dataloader 的，具体 num_workers 是 dataloader 自己去处理的。sampler 只需关心当前worker 需要需处理哪些 sample 的数据
    里面就是按照 (global) batch size = data_parallel_world_size()[2] * micro_batch_size 来取数据，然后只关注自己的那部分(rank*micro_batch_size, iter*global_batch_size)，然后 yield。所以是处理好每个micro batch，然后在里面划分data sharding
    
    _iter__(self): 返回的是一个 micro batch size 大小的采样数据
    
```
random 时，不需要 多个 进程之间用同一个seed嘛？需要，所以里面的 seed 统一用 self.epoch 来做。

```
MegatronPretrainingRandomSampler

```

## training.py:

## 问题
1. 如何实现 Pipieline 里的 interleaving 1F1B 调度？
2. tensor model parallel 里如何同步的？只是 g 和 f 那里需要同步把？
3. data parallel group 里，是只要对应颜色(2个)需要同步？还是说4个之间的 all-reduce？应该是后者，因为有 tensor model parallel，所以只需要同步一个 tensor 的相同 split 部分。如果模型不是横向都切分，只切分了大的 op，那就得升级 mapping 关系了
4. Megatron-LM 是否可以用于 CV 模型？是否是特化的。特化于 transformer 的，为何不能用于 CV ？
6. seq-length 这些参数如何起作用的？ 
7. schedules.py 里的 model_trunk 干嘛的？
8. 这些操作语义都清楚了，但是是否并发，如何同步？

## 可优化的方向
1. 目前还有重复计算的地方，比如 Column Parallel Linear 里， split 逻辑在多个 tensor model parall 的ranks 上运行，数据也是全量的
