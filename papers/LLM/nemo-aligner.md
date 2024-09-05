PPO Training

PPO Inference:

Features:
* TRT-LLM 推理引擎
* Engine update with Refit
* Worker Pool for Load Balance

由于 reference policy 和 actor 是相同的模型，不用的权重。所以我们把他们结合到一个 job 里，然后 offload reference policy 的权重到 cpu 上，这样可以和 actor 的权重进行 swap。同样的策略也用到了 reward 模型和 critic 上。

所有的通信都是异步的，让 critic inference/training 和 policy inference/training 流水起来

我们扩展计算上分配的 size，让 :
1. 推理时：reward model 推理+critic 推理大约等于 actor sampling + reference policy 推理
2. critic train <= actor train + actor inference initialization(这是什么阶段？指更新权重)


### 3.4 Scalability
在 Llama 2 70B actor 和 Llama 2 13B critic 上测试的，8 actor nodes，2 critic node。然后扩大2倍、3倍的节点。

Train 阶段的 scaling 是亚线性的(1.69, 2.5x)。因为每个 dp rank 上的 micro batch 会在节点数量增加后降低。因为在 PP 并行场景下，optimizer 执行之前必须把所有 pipeline stages 都结束了，所以会有一个填充和消耗完 pipeline 的开销，无论 micro-batch 的大小。因此当减小 microbatch 大小时，增大了填充和消耗 pipeline 的开销，因此 GPU 利用率会很差。

在 Rollout stage，Response generation 和 Log-probs 的计算能够随着节点数增加而很好地扩展。因为这里 dp 增加，大家都均分任务。

Refit 的主要耗时是在 reload engine（即读取），所以随着节点数增加，并不会显著增大。