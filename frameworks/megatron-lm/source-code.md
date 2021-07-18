tensor model parallel + pipeline model parallel 的例子在：
*\_distributed\_with\_mp.sh 里

pipeline model parallel 参数：
pipeline-model-parallel-size 2

split 949,50,1 // 数据划分为三个：training/validation/test 集合
global-batch-size 16
micro batch size 4


tensor model parallel 参数：


## TODO
1. 看如何用几行代码实现的 tensor 级别并行，而且是在 g 和 f 那里要all-reduce 一下

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

## 调度? schedules.py
forward step 里，自己这里完了后，把输出结果发给下游
`p2p_communication.send_forward()`
    forward_backward_pipelining_without_interleaving
    
    
## 问题
1. 如何实现 Pipieline 里的 interleaving 1F1B 调度？
2. tensor model parallel 里如何同步的？只是 g 和 f 那里需要同步把？
3. data parallel group 里，是只要对应颜色(2个)需要同步？还是说4个之间的 all-reduce？
