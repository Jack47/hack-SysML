1. 在解决的是什么问题？ 作为集群调度系统，首次考虑了训练过程的统计学效率，同时兼顾了资源分配和参数的选择。最终目标是优化 goodput
2. 为何成功，标志/准是什么？ 训练速度平均减少了 37-50%。高了集群的 goodput  
3. 在前人基础上的关键创新是什么？ 定义了集群调度的 goodput，首次考虑训练过程的统计学效率
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 问题
1. 看PolluxAgent 如何 profile pytorch 里运行程序的速度(iteration speed)的? 看看是否对程序有侵入性，这都是两个程序的行为了，应该是用的 profile agent

## 疑问
1. 为什么做了调度上的优化后，训练速度反而快了（训练效率更高）
2. 动态增减资源后， DDP 下需要如何适配？类似 ElasticDL 一样，需要重启，大家用新的 world_size 来训练？
4. 它用了什么预测模型？输入有哪些，输出有哪些？需要多少数据
5. 看看 xiaowencong 的 AntMan 有开源嘛？

## 实现
PolluxAgent：Profile 和 Tuner & LRScaler
PolluxSched：Model of jobs goodputs, Dynamically re-allocate resources

AdaptDL 包含一个 K8S 服务形式的 job 调度器，还有一个自适应的 训练库(adaptdl.torch.AdaptiveDataParallel, AdaptiveDataLoader)

目测上述两个应该是有一定的动态能力？


### init_process_group
只不过是从 url 里获得：

key=adaptdl.env.job_id()
group=adaptdl.env.num\_restarts() # 重启次数？

url/discover/key/group # 这样也挺合理，相当于 job_id 标识一个训练任务，而多次动态调整期间，这个 id 应该不变。变的是 num\_restarts，每次
重启后，重新组成一个新拓扑。

我们做 gpus，也可以这样呀，封装一个简单的 lib 函数，让用户用我们的这个库替换就行？

### [supervisor](https://github.com/petuum/adaptdl/blob/11dd3ad691f89a9f02282737ce2a57015f0d3349/sched/adaptdl_sched/supervisor.py?plain=1#L65)
找出标有 adaptdl/group 为同一个 job 的，而且同一个 group的组，返回 pod\_ip\_list。

所以就是通过一个中心化服务，大家都带着一些标签来做这个事儿。最重要的是知道 master ip：取list里第一个

## 启发
1. 通过调度，能让训练更快
2. 可以整一个预测模型，预测某任务在不同硬件和参数下，速度是多少？那我们是否也可以这样来加速训练
3. 增加 batch size，对最终精度提高有帮助，但是训练效率会变慢？ On the computational inefficiency of large batch sizes for stochastic gradient descent.
4. 他们测试的方法是把 Azure 上任务的 trace 结果拿出来，在一个实验台性质的集群里回放，然后统计结果
5. gradient noise scale: GNS, 用来说明训练效率对于大 batchsize 和 LR 是否敏感
6. 可以看看实验台上的日志和统计分析脚本：pollux-results.
7. 我们也可以实现这样一个动态改变 worker 数量的训练系统，用空闲资源训练任务？里面 AdaptiveDataloader，ElasticDataSampler, checkpoint 和 动态 nccl 都做好了

## 看代码的启发
[import portpicker](https://github.com/petuum/adaptdl/blob/11dd3ad691f89a9f02282737ce2a57015f0d3349/adaptdl/adaptdl/torch/__init__.py?plain=1) portpicker.pick_\unused\_port()

