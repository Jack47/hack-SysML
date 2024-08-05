# Ray-vLLM

type: Post
status: Published
tags: machine system
category: 深度学习

# 作业

**搞明白ray的基础原理和用法，出一个文档介绍，着重介绍下利用ray如何部署 vLLM？比如怎么启动ray，怎么启动多个vLLM实例，多个dp_rank怎么调用它**

## ray部署前置条件

1. **所有集群设备必须能相互通信**
    1. 在同一局域网下
    2. 个人主机<->云服务器
2. **运行环境与依赖项一致，可以使用anaconda / Docker来保证**
    1. 使用虚拟环境或Anaconda
        1. **在头节点创建环境**：
           
            ```bash
            conda create -n myenv python=3.10
            conda activate myenv
            pip install -r requirements.txt
            ```
            
        2. **导出环境**：
           
            ```bash
            conda env export > environment.yml
            ```
            
        3. **在工作节点上创建相同环境**：
           
            ```bash
            conda env create -f environment.yml
            conda activate myenv
            ```
        
    2. 使用Docker
        1. **创建Dockerfile**：
           
            ```
            FROM python:3.10
            COPY requirements.txt .
            RUN pip install -r requirements.txt
            ```
            
        2. **构建Docker镜像**：
           
            ```bash
            docker build -t myrayimage .
            ```
            
        3. **在所有节点上运行Docker容器**：
           
            ```bash
            docker run -it --network host --name myraycontainer myrayimage
            ```
            
    
    ## ray连接集群
    
    1. 选择一个并行设备作为头结点，负责训练文件的编辑和运行，以及其他工作如调试
       
        在命令行输入：`ray start --head` 
        
    2. 其他节点作为普通节点，主要负责提供算力运行头节点的代码
       
        在命令行输入：`ray start --address="<head_node_ip>:6379"`
        
    3. 在完成上述工作之后，可以输入 `ray status` 查看各个设备的情况
    4. 建议不要通过 `python x.py` 开始任务，而是通过`ray cli`的`ray job submit`来提交作业，这样可以省去在脚本中手动初始化和配置 Ray 集群，同时还支持对作业进行跟踪、查看状态和获取日志等操作
    
    ## ray训练基本用法
    
    ## **入门用法**
    
    - **Scaling configuration**
        - `scaling_config = ScalingConfig(num_workers=1, use_gpu=**False**)`
        - **worker**
            - 即进程数，决定并发度
    - **Trainer**
      
        ```python
        from ray.train.torch import TorchTrainer
        from ray.train import ScalingConfig
        
        def train_func():
            # Your PyTorch training code here.
            ...
        
        scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
        trainer = TorchTrainer(train_func, scaling_config=scaling_config)
        result = trainer.fit()
        ```
        
        - `train_func`**:**
            - 模型训练代码，一般是加载数据集、训练保存模型、记录指标的函数
        - `scaling_config`:
            - 缩放设置，包含进程数和是否使用gpu
    - **Set up a training function**
      
        ```python
        def train_func(config):
            lr = config["lr"]
            num_epochs = config["num_epochs"]
        
        config = {"lr": 1e-4, "num_epochs": 10}
        trainer = ray.train.torch.TorchTrainer(train_func, train_loop_config=config, ...)
        ```
        
        - 可以使用config_dict作为参数输入训练函数
        - 对于大的数据类型，直接在train_func初始化，不要传递
    - **Set up a model**
      
        ```python
        def train_func():
             ...
             # Create model.
             model = ...
             ...
             model = ray.train.torch.prepare_model(model)
        ```
        
        - 不再需要指定设备和使用torch的分布式函数，直接调用ray的函数
    - **Set up a dataset**
      
        ```python
         def train_func():
        
             ...
        
             dataset = ...
        
             data_loader = DataLoader(dataset, batch_size=worker_batch_size, shuffle=True)
             data_loader = ray.train.torch.prepare_data_loader(data_loader)
        
             for epoch in range(10):
                 if ray.train.get_context().get_world_size() > 1:
                     data_loader.sampler.set_epoch(epoch)
        ```
        
        - 同样的不需要指定设备，使用ray提供的data_loader
        - `ray.train.get_context()`：获取当前训练的上下文（context）。
        - `get_world_size()`：返回当前训练任务中并行运行的工作进程（worker）的数量，也称为“世界大小”（world size）。
        - `if ray.train.get_context().get_world_size() > 1`：判断是否在分布式训练环境中运行。如果 `world_size` 大于 1，表示有多个工作进程在并行训练
        - `data_loader.sampler.set_epoch(epoch)`：这使得每个并行的进程处理的数据分片在周期内随机
    - **Report checkpoints and metrics**
      
        ```python
        import os
        import tempfile
        
        import ray.train
        
         def train_func():
        
             ...
        
             with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt")
                )
        
                metrics = {"loss": loss.item()}  # Training/validation metrics.
        
                # Build a Ray Train checkpoint from a directory
                checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
        
                # Ray Train will automatically save the checkpoint to persistent storage,
                # so the local `temp_checkpoint_dir` can be safely cleaned up after.
                ray.train.report(metrics=metrics, checkpoint=checkpoint)
        
             ...
        ```
        
        - ray.train.report可以自动记录指标、保存模型
    - **Configure persistent storage**
      
        ```python
        from ray.train import RunConfig
        
        # Local path (/some/local/path/unique_run_name)
        run_config = RunConfig(storage_path="/some/local/path", name="unique_run_name")
        
        # Shared cloud storage URI (s3://bucket/unique_run_name)
        run_config = RunConfig(storage_path="s3://bucket", name="unique_run_name")
        
        # Shared NFS path (/mnt/nfs/unique_run_name)
        run_config = RunConfig(storage_path="/mnt/nfs", name="unique_run_name")
        ```
        
        - 这是保存路径，支持本地路径、共享云存储和网络文件路径
    - **Access training results**
      
        ```python
        result.metrics     # The metrics reported during training.
        result.checkpoint  # The latest checkpoint reported during training.
        result.path        # The path where logs are stored.
        result.error       # The exception that was raised, if training failed.
        result.best_checkpoint
        ```
        
    
    ## 完整指南
    
    ### 数据加载和预处理
    
    1. 创建ray数据集
       
        ```python
        
        train_data = ray.data.read_csv("./train.csv")
        train_data = ray.data.from_huggingface(hf_train_ds)
        ```
        
    2. 预处理
       
        ```python
        # increment是数据预处理函数
        train_dataset = train_dataset.map_batches(increment)
        ```
        
    3. 在train_func中获取共享数据集
       
        ```python
        train_data_shard = train.get_dataset_shard("train")
        train_dataloader = train_data_shard.iter_torch_batches(
            batch_size=batch_size, dtypes=torch.float32
        )
        ```
        
    4. 在Trainer中传入数据集
       
        ```python
        trainer = TorchTrainer(
            train_func,
            datasets={"train": train_dataset},
            scaling_config=ScalingConfig(num_workers=2, use_gpu=use_gpu)
        )
        ```
        
    
    ### 配置进程数和GPU
    
    - 在ScalingConfig中配置进程数和是否使用gpu
    - 为进程分配资源: `resources_per_worker`
    - 设置GPU类型
    - 设置通信后端
      
        ```python
        trainer = TorchTrainer(
            train_func,
            scaling_config=ScalingConfig(
                num_workers=8,
                use_gpu=True,
                resources_per_worker={
        	        "CPU": 4,
        	        "GPU": 0.5,
                },
                accelerator_type="A100"
            ),
            torch_config=TorchConfig(backend="gloo"),
        )
        ```
        
    - 设置网络接口
      
        ```python
        import ray
        
        runtime_env = {"env_vars": {"NCCL_SOCKET_IFNAME": "ens5"}}
        ray.init(runtime_env=runtime_env)
        ```
        
    
    ### 配置存储
    
    **Ray Train 希望所有工作线程都能够将文件写入同一持久存储位置**
    
    - 在`RunConfig`中配置
    
    ```python
    path = "/tmp/custom/storage/path" #本地，单节点集群
    path = "/mnt/cluster_storage" #共享文件
    path = "s3://bucket-name/sub-path/" #云存储
    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_path=path,
            name="experiment_name",
        )
    )
    ```
    
    ### 监控和指标记录
    
    Ray Train 原生支持 [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/)
    
    ```python
    from ray import train
    def train_func(config):
    	...
    	valid_loss = valid_loss.item()
      mape_collected = mape.compute().item()
      mean_valid_loss_collected = mean_valid_loss.compute().item()
    
      train.report(
          {
              "mape_collected": mape_collected,
              "valid_loss": valid_loss,
              "mean_valid_loss_collected": mean_valid_loss_collected,
          }
      )
    ```
    
    ### 保存与加载检查点
    
    1. **存储性能最佳的模型权重：**将您的模型保存到持久性存储中，并将其用于下游服务/推理。
    2. **容错：**在可抢占的计算机/Pod 集群上长时间运行的训练作业中处理节点故障。
    3. **分布式检查点：**在进行*模型并行训练*时，Ray Train 检查点提供了一种简单的方法，可以[并行上传每个工作线程的模型分片](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html#train-distributed-checkpointing)。 无需将完整模型收集到单个节点。
    4. **与 Ray Tune 集成：**某些 [Ray Tune 调度器](https://docs.ray.io/en/latest/tune/api/schedulers.html#tune-schedulers)需要保存和加载检查点。
    
    ```python
    import tempfile
    
    from ray import train
    
    def train_fn(config):
        ...
    
        metrics = {...}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
    
            # Only the global rank 0 worker saves and reports the checkpoint
            if train.get_context().get_world_rank() == 0:
                ...  # Save checkpoint to temp_checkpoint_dir
    
                checkpoint = Checkpoint.from_directory(tmpdir)
    
            train.report(metrics, checkpoint=checkpoint)
    ```
    
    - temp_checkpoint_dir是你本地暂存模型的目录
    - train.report将数据提交到长期存储路径，即云/共享存储
    - 同时还可以配置保存检查点的属性
      
        ```python
        from ray.train import RunConfig, CheckpointConfig
        
        # Example 1: Only keep the 2 *most recent* checkpoints and delete the others.
        run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=2))
        
        # Example 2: Only keep the 2 *best* checkpoints and delete the others.
        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                # *Best* checkpoints are determined by these params:
                checkpoint_score_attribute="mean_accuracy",
                checkpoint_score_order="max",
            ),
            # This will store checkpoints on S3.
            storage_path="s3://remote-bucket/location",
        )
        ```
        
    
    ### 超参数优化
    
    - 直接使用Tuner来调参
      
        ```python
        import ray
        from ray import train, tune
        from ray.tune import Tuner
        from ray.train.xgboost import XGBoostTrainer
        
        dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")
        
        trainer = XGBoostTrainer(
            label_column="target",
            params={
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "error"],
                "max_depth": 4,
            },
            datasets={"train": dataset},
            scaling_config=train.ScalingConfig(num_workers=2),
        )
        
        # Create Tuner
        tuner = Tuner(
            trainer,
            # Add some parameters to tune
            param_space={"params": {"max_depth": tune.choice([4, 5, 6])}},
            # Specify tuning behavior
            tune_config=tune.TuneConfig(metric="train-logloss", mode="min", num_samples=2),
        )
        # Run tuning job
        tuner.fit()
        ```
        
    - Tuner配置
      
        ```python
        from ray.tune import TuneConfig
        from ray.tune.search.bayesopt import BayesOptSearch
        
        tune_config = TuneConfig(
            metric="loss",
            mode="min",
            max_concurrent_trials=10,
            num_samples=100,
            search_alg=BayesOptSearch(),
        )
        ```
        
    
    
    
    ## ray部署vLLM实例
    
    (交换分区所需内存过大跑不了，但是好像没别的报错？)
    
    ```python
    import ray
    from vllm import LLM, SamplingParams
    
    @ray.remote
    class vLLMEngine:
        def __init__(self, model_name, tensor_parallel_size):
            self.model_name = model_name
            self.tensor_parallel_size = tensor_parallel_size
            self.load_model()
    
        def load_model(self):
            print(f"Loading model {self.model_name} with tensor_parallel_size {self.tensor_parallel_size}")
            self.llm = LLM(self.model_name, tensor_parallel_size=self.tensor_parallel_size)
    
        def generate_text(self, prompt):
            print(f"Generating text for prompt: {prompt}")
            sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
            result = self.llm.generate(prompt, sampling_params=sampling_params)
            return result.generations[0].text
    
    @ray.remote
    class DPRank:
        def __init__(self, rank, vllm_engine):
            self.rank = rank
            self.vllm_engine = vllm_engine
    
        def train_step(self, data):
            generated_text = ray.get(self.vllm_engine.generate_text.remote(data))
            return f"Rank {self.rank} trained with data: {generated_text}"
    
    ray.init()
    
    vllm_engines = [vLLMEngine.remote('EleutherAI/gpt-neo-2.7B', tensor_parallel_size=4) for _ in range(4)]
    dp_ranks = [DPRank.remote(rank, vllm_engines[rank % len(vllm_engines)]) for rank in range(4)]
    
    prompts = ["Hello, world!", "How are you?", "This is a test.", "What is your name?"]
    futures = [engine.generate_text.remote(prompt) for engine, prompt in zip(vllm_engines, prompts)]
    results = ray.get(futures)
    print(results)
    ```
    



**以上为作业部分**
**下面是完成作业所收集的资料（说实话主要有用的还是ray和vllm的文档）**

------

----





























# 学习资料

## Ray

- [https://www.jiqizhixin.com/articles/2023-01-03-2](https://www.jiqizhixin.com/articles/2023-01-03-2)
    - 未来ai应用程序的形态：与环境存在连续的交互，并从交互动作中进行学习。这些应用必然越来越多地在动态环境中来完成任务，根据环境的变化作出反应，并执行一系列的动作来达到长期目标。这些特性对于运行环境性能和灵活性等方面提出了全新且苛刻的系统要求，因此研究者提出了基于分布式的 Ray 框架
    - 作为一个分布式计算框架，Ray 有两个关键优势，分别是位置感知（Locality-aware）和任务分配（task placement）。Ray 能够横向扩展系统，以支持高吞吐量的细粒度任务，同时保持容错和低延迟任务调度
- https://arxiv.org/abs/1712.05889
- [https://github.com/ray-project/ray](https://github.com/ray-project/ray)
- [https://ray.osanswer.net/t/topic/448](https://ray.osanswer.net/t/topic/448)
- [**https://github.com/ray-project/ray-educational-materials?tab=readme-ov-file](https://github.com/ray-project/ray-educational-materials?tab=readme-ov-file) *****

## RL

### [人工智能 - 一文读懂强化学习：RL全面解析与Pytorch实战 - 个人文章 - SegmentFault 思否](https://segmentfault.com/a/1190000044356457)

## **引言**

- 强化学习没有标签，依靠智能体(Agent)通过不断试错、适应和优化学习来学习如何在给定环境中实现特定目标
- 核心组成
    - 状态（state）：反应环境或系统目前的状态
    - 动作（action）：智能体在特定状态下可以采取的操作
    - 奖励（reward）：一个数值反馈，用于量化智能体采取某一动作后环境的反应
    - 策略（policy）：一个映射函数，指导智能体在特定状态下应采取哪一种动作（状态机？）
- 强化学习能不断适应变化的环境或参数，在自动决策系统中的应用越来越广泛

## 强化学习基础

强化学习的核心是建模决策问题，通过与环境的交互来学习最佳的决策方法，这一过程通常是通过马尔科夫决策过程来实现的

- **马尔可夫决策过程（MDP）**
  
- MDP是用来描述决策问题的数学模型，主要由一个四元组 ( (S, A, R, P) ) 组成。
    - **状态空间（S）**: 表示所有可能状态的集合。
    - **动作空间（A）**: 表示在特定状态下可能采取的所有动作的集合。
    - **奖励函数（R）**: ( R(s, a, s') ) 表示在状态 ( s ) 下采取动作 ( a ) 并转移到状态 ( s' ) 时所获得的即时奖励。
    - **转移概率（P）**: ( P(s' | s, a) ) 表示在状态 ( s ) 下采取动作 ( a ) 转移到状态 ( s' ) 的概率。
    
    ### **状态（State）**
    
    在MDP中，状态是用来描述环境或问题的现状。在不同应用中，状态可以有很多种表现形式：
    
    - 在棋类游戏中，状态通常表示棋盘上各个棋子的位置。
    - 在自动驾驶中，状态可能包括车辆的速度、位置、以及周围对象的状态等。
    
    ### **动作（Action）**
    
    动作是智能体（Agent）在某一状态下可以采取的操作。动作会影响环境，并可能导致状态的转变。
    
    - 在股市交易中，动作通常是“买入”、“卖出”或“持有”。
    - 在游戏如“超级马里奥”中，动作可能包括“跳跃”、“下蹲”或“向前移动”等。
    
    ### **奖励（Reward）**
    
    奖励是一个数值反馈，用于评估智能体采取某一动作的“好坏”。通常，智能体的目标是最大化累积奖励。
    
    - 在迷宫问题中，到达目的地可能会得到正奖励，而撞到墙壁则可能会得到负奖励。
    
    ### **策略（Policy）**
    
    策略是一个从状态到动作的映射函数，用于指导智能体在每一状态下应采取哪一动作。形式上，策略通常表示为 ( \pi(a|s) )，代表在状态 ( s ) 下采取动作 ( a ) 的概率。
    
    - 在游戏如“五子棋”中，策略可能是一个复杂的神经网络，用于评估每一步棋的优劣。
    
    通过优化策略，我们可以使智能体在与环境的交互中获得更高的累积奖励，从而实现更优的性能
    

## 常用强化学习算法

### 值迭代

基于动态规划来计算最优策略，主要用于解决具有完全可观测状态和已知转移概率的MDP问题，是一种“模型已知”的算法，应用于路径规划、迷宫问题的等环境中

### Q学习

基于值函数的“模型无知”算法，通过更新Q值找到最优策略？

### 策略梯度

在策略空间进行优化，通过计算梯度来更新策略参数，适合处理高维或连续的动作和空间，应用于自然语言处理、连续控制问题

### Actor-Critic(演员-评论家)

Actor-Critic 结合了值函数方法和策略梯度方法的优点。其中，"Actor" 负责决策，"Critic" 负责评价这些决策，广泛应用在在自动驾驶、资源分配和多智能体系统等复杂问题中

### **PPO（Proximal Policy Optimization）**

PPO是一种高效、可靠的强化学习算法，属于策略梯度家族的一部分

**原理**

PPO的核心思想是通过限制策略更新的步长来避免太大的性能下降。这是通过引入一种特殊的目标函数实现的，该目标函数包含一个剪辑（Clipping）项来限制策略的改变程度

目标函数如下：

![Untitled](Ray%E5%AD%A6%E4%B9%A0_openrlhf%E6%A8%A1%E5%9E%8B%20f8f7edcbbd834b8c9bbc88d101aece9d/Untitled%204.png)

**细节**

- **多步优势估计**: PPO通常与多步回报（Multi-Step Return）和优势函数（Advantage Function）结合使用，以减少估计误差。
- **自适应学习率**: PPO通常使用自适应学习率和高级优化器（如Adam）。
- **并行采样**: 由于PPO是一种“样本高效”的算法，通常与并行环境采样结合使用，以进一步提高效率。

**代码**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.policy_head = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.softmax(self.policy_head(x), dim=-1)

# 初始化
state_dim = 4  # 状态维度
action_dim = 2  # 动作维度
policy_net = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
epsilon = 0.2

# 采样数据（这里假设有一批样本数据）
states = torch.rand(10, state_dim)
actions = torch.randint(0, action_dim, (10,))
advantages = torch.rand(10)

# 计算旧策略的动作概率
with torch.no_grad():
    old_probs = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze()

# PPO更新
for i in range(4):  # Typically we run multiple epochs
    action_probs = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze()
    ratio = action_probs / old_probs
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("PPO Update Done!")
```

**文章**

[https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f](https://www.notion.so/eb7b2d1891f44b3a84e7396d19d39e6f?pvs=21)

## Ray

## 原理

### background

在LLM对巨大数据和算力的驱使下，对大数据处理框架提出了新的需求

传统的大数据处理框架(如Hadoop和Spark)虽然提供了分布式数据处理能力，但是灵活性和实时性无法满足需求

现有深度学习框架虽然支持分布式训练，但是使用门槛高，对开发者不友好，且不同任务需要对应不同的编程模型

现代计算集群可能由多种计算资源组成(CPU、GPU、TPU),如何高效利用这些异构资源也是重要课题

## motivation

Ray的目标就是解决上述问题一，提供一个统一的编程模型，满足大规模并行计算、分布式计算和异构资源管理，主要目标包括：

1. 简化分布式计算编程：提供有好的api，使开发者能快速将单机应用拓展为分布式应用，而无需了解底层分布式系统的复杂性
2. 高效的资源管理：支持多种类型的计算资源(CPU、GPU),并提供高效的资源调度和管理机制，以最大化计算资源的利用率
3. 灵活的任务调度：支持动态的任务调度和负载均衡，以适应不同计算任务和工作负载
4. 集成和互操作性：与现有的大数据框架、机器学习框架无缝集成，提供丰富的生态支持

## 架构

[**Ray**](https://www.ray.io/) 是一个开源的统一计算框架，使得扩展 AI 和 Python 工作负载变得容易。

[**Ray 集群**](https://docs.ray.io/en/latest/cluster/getting-started.html) 是一组连接到共同 Ray 头节点的工作节点。Ray 集群可以是固定大小的，也可以根据在集群上运行的应用程序请求的资源自动缩放。

[**Ray Core**](https://docs.ray.io/en/latest/ray-core/walkthrough.html) 是一个开源的、Python 的、通用的分布式计算库，使 ML 工程师和 Python 开发者能够加速机器学习工作负载并扩展 Python 应用程序。

[**Ray AI Runtime (AIR)**](https://docs.ray.io/en/latest/ray-air/getting-started.html) 是一个开源的、Python 的、特定领域的库，为 ML 工程师、数据科学家和研究人员提供了一个可扩展和统一的工具包，用于 ML 应用程序。

## AIR

[Ray 的](https://docs.ray.io/en/latest/ray-air/getting-started.html)五个本机库各自分配一个特定的 ML 任务：

- [数据](https://docs.ray.io/en/latest/data/data.html)：可扩展、与框架无关的数据加载和转换，涵盖训练、调整和预测。
- [训练](https://docs.ray.io/en/latest/train/train.html)：具有容错功能的分布式多节点和多核模型训练，并与流行的训练库集成。
- [调整](https://docs.ray.io/en/latest/tune/index.html)：可扩展的超参数调整以优化模型性能。
- [服务](https://docs.ray.io/en/latest/serve/index.html)：可扩展且可编程的服务，用于部署模型进行在线推理，并具有可选的微批处理来提高性能。
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html)：可扩展的分布式强化学习工作负载。

ML pracitioners tend to run into a few common problems with training models that prompt them to consider distributed solutions:

1. Training time is [too long](https://www.anyscale.com/blog/how-anastasia-implements-ray-and-anyscale-to-speed-up-ml-processes-9x) to be practical.
2. The [data is too large](https://www.anyscale.com/blog/how-ray-and-anyscale-make-it-easy-to-do-massive-scale-machine-learning-on) to fit on one machine.
3. [Training many models](https://www.anyscale.com/blog/training-one-million-machine-learning-models-in-record-time-with-ray) sequentially doesn't utilize resources efficiently.
4. The [model itself is too large](https://www.uber.com/blog/horovod-ray/) to fit on a single machine.

[Ray Train](https://docs.ray.io/en/latest/ray-air/trainer.html) addresses these issues by improving performance through distributed multi-node training.

# Ray Train

## hf-transformers

比起原本的pytorch+transformers代码，增加了以下部分

- import
  
    ```python
    import ray.train.huggingface.transformers
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer
    ```
    
- callback
  
    ```python
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    ```
    
- prepare_trainer
  
    ```python
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    ```
    
- launch train_func
  
    ```python
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        # [4a] If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # run_config=ray.train.RunConfig(storage_path="s3://..."),
    )
    ```
    
- full code
  
    ```python
    import os
    
    import numpy as np
    import evaluate
    from datasets import load_dataset
    from transformers import (
        Trainer,
        TrainingArguments,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    
    import ray.train.huggingface.transformers
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer
    
    # [1] Encapsulate data preprocessing, training, and evaluation
    # logic in a training function
    # ============================================================
    def train_func():
        # Datasets
        dataset = load_dataset("yelp_review_full")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
    
        small_train_dataset = (
            dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
        )
        small_eval_dataset = (
            dataset["test"].select(range(1000)).map(tokenize_function, batched=True)
        )
    
        # Model
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=5
        )
    
        # Evaluation Metrics
        metric = evaluate.load("accuracy")
    
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
    
        # Hugging Face Trainer
        training_args = TrainingArguments(
            output_dir="test_trainer",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            report_to="none",
        )
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )
    
        # [2] Report Metrics and Checkpoints to Ray Train
        # ===============================================
        callback = ray.train.huggingface.transformers.RayTrainReportCallback()
        trainer.add_callback(callback)
    
        # [3] Prepare Transformers Trainer
        # ================================
        trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    
        # Start Training
        trainer.train()
    
    # [4] Define a Ray TorchTrainer to launch `train_func` on all workers
    # ===================================================================
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        # [4a] If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # run_config=ray.train.RunConfig(storage_path="s3://..."),
    )
    result: ray.train.Result = ray_trainer.fit()
    
    # [5] Load the trained model.
    with result.checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = os.path.join(
            checkpoint_dir,
            ray.train.huggingface.transformers.RayTrainReportCallback.CHECKPOINT_NAME,
        )
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    ```
    

  

# Ray Cluster

Ray 集群由单个[头节点](https://docs.ray.io/en/latest/cluster/key-concepts.html#cluster-head-node)和任意数量的已连接[工作节点](https://docs.ray.io/en/latest/cluster/key-concepts.html#cluster-worker-nodes)组成

### 同一局域网内

为保证运行环境一致(Docker or Anconda)，最好在所有设备上开启相同的虚拟环境、有条件的统一管理的话直接让所有设备使用同一个虚拟环境，进入虚拟环境：

**init** : `pip install "ray[default]"`

1. 确定一台头节点设备，输入命令以启动该头接待你
   
    `ray start --head --port=6379`
    
2. 对于其他设备，输入命令由此连接到头节点
   
    `ray start --address=<head-node-address:port>`
    
    其中`head-node-address`是头节点ip地址
    
3. 在所有设备上输入ray status查看连接情况

### 不在同一局域网内

**Kubernetes部署：……**

**本地与云服务器连接**：

`curl ifconfig.me` 输出该服务器的公网ip，使用公网ip作为address

**两台主机的WSL之间：**

可能需要桥接

# OpenRLHF

## 架构

### Actor Model (vLLM Inference)

**Actor Model** 是指一种并行计算模型，其中计算单元被称为“Actors”。每个 Actor 是一个独立的计算实体，拥有自己的状态和行为，可以与其他 Actors 进行异步通信。在 vLLM（虚拟大语言模型）推理中，Actor 模型用于并行化和分布推理任务，以提高推理效率和吞吐量。

- **作用**：在分布式环境中，每个 Actor 可以独立地处理一部分推理任务，彼此之间通过消息传递进行通信。这种模型适用于需要高并发和低延迟的大规模推理任务。

### Reference Model

**Reference Model** 通常是指在强化学习或监督学习中用作基准的模型。它可以是用于比较的已知性能模型，或者在策略改进过程中用作参考的目标模型。

- **作用**：在强化学习中，Reference Model 可以作为当前策略模型（Policy Model）进行比较和评估，以帮助调整和优化策略。在监督学习中，它可能是一个已经训练好的模型，用于评估新模型的性能。

### Actor Model

除了在推理任务中的应用外，Actor Model 还广泛应用于强化学习中的策略模型。强化学习中的 Actor-Critic 方法中，Actor 负责选择动作并与环境交互，以最大化累积奖励。

- **作用**：在强化学习中，Actor 负责根据当前策略选择动作，并与环境进行交互，以收集反馈（奖励和状态变化）。Actor 的目标是通过与环境的交互，不断改进其策略，以最大化长期回报。

### Reward Model

**Reward Model** 在强化学习中用于评估和计算每个动作的奖励。它根据环境的反馈，为每个动作分配一个奖励值，这个奖励值用于指导策略的改进。

- **作用**：在训练过程中，Reward Model 根据当前状态和采取的动作，计算并返回相应的奖励。这些奖励信息被用来更新策略模型（Actor），以提高模型在不同状态下的决策质量。

### Critic Model

**Critic Model** 是 Actor-Critic 方法中的另一个核心组件。它负责评估当前策略的表现，并为 Actor 提供策略改进的反馈。Critic 通过估计状态值（Value Function）来评估当前策略的优劣。

- **作用**：在强化学习中，Critic 评估给定状态的预期回报（即状态值），并根据实际奖励和预期回报之间的差距，计算策略的改进方向。这些信息用于指导 Actor 更新策略，以提高整体策略的有效性

## trainer

### ray

- `lancher.py`
    - `class DistributedTorchRayActor`
        - 初始化Ray的分布式环境变量。
        - 提供获取当前节点IP和空闲端口的功能。
        - 设置分布式训练所需的环境变量，如 `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE` 和 `RANK`
    - **`BasePPORole`**
        - `@ray.remote(num_gpus=1)`
        - 继承 `DistributedTorchRayActor`，添加分布式策略的设置功能。
        - 提供一个未实现的 `init_model_from_pretrained` 方法，用于初始化预训练模型。
    - **`ReferenceModelRayActor`**
        - `@ray.remote(num_gpus=1)`
        - 继承 `BasePPORole`，实现引用模型的初始化和推理功能。
        - 使用 `DeepspeedStrategy` 进行模型分布式设置。
        - 定义 `forward` 方法用于模型的前向推理。
        - 定义 `empty_cache` 方法清理GPU缓存。
    - **`RewardModelRayActor`**
        - `@ray.remote(num_gpus=1)`
        - 继承 `BasePPORole`，实现奖励模型的初始化和推理功能。
        - 使用 `DeepspeedStrategy` 进行模型分布式设置。
        - 定义 `forward` 方法用于模型的前向推理。
        - 定义 `empty_cache` 方法清理GPU缓存。
    - **`PPORayActorGroup`**
        - `__init__`: 初始化PPORayActorGroup类，设置节点数量、每个节点的GPU数量、Actor类型等。
        - `_initiate_actors`: 根据节点和GPU数量初始化Actor，支持使用Ray的Placement Group锁定资源。
        - **异步方法**
            - `async_init_model_from_pretrained`: 异步初始化预训练模型。
            - `async_fit_actor_model`: 异步训练Actor模型。
            - `async_save_model`: 异步保存模型。
            - `async_run_method`: 异步运行指定方法。
- `ppo_actor.py`
    - `class ActorPPOTrainer(PPOTrainer)`
        - **初始化**:
            - `vllm_engines`：可选的vLLM引擎列表。
            - `remote_rm_url`：远程资源管理器URL列表。
            - `critic_train_remote`：是否在远程训练评论者模型。
            - 创建 `RemoteExperienceMaker` 对象，用于生成和管理经验数据。
        - **初始化分布式进程组**:
            - 如果有 `vllm_engines`，则初始化分布式进程组。
            - 获取主节点地址和端口。
            - 设置分布式训练的后端（默认是 `nccl`，如果 vLLM 版本大于0.4.2，则使用 `gloo`）。
            - 调用每个 vLLM 引擎的 `init_process_group` 方法，初始化进程组。
            - 等待所有进程组初始化完成。
    - `class ActorModelRayActor(BasePPORole)`
        - **初始化模型**
            - 使用 `DeepspeedStrategy` 设置分布式训练
            - 初始化演员模型 `Actor`
            - 创建并设置 `tokenizer`
            - 如果启用了EMA（指数移动平均），则初始化EMA模型
            - 创建优化器 `actor_optim`
            - 准备数据集 `prepare_datasets`
            - 设置调度器 `actor_scheduler`
            - 启用梯度检查点
            - 准备模型、优化器和调度器
            - 加载检查点
        - `prepare_datasets`
            - 混合和选择提示数据，创建 `PromptDataset` 和相应的数据加载器。
            - 如果存在预训练数据，则混合和选择预训练数据，创建 `SFTDataset` 和相应的数据加载器
        - `fit`
            - 创建 `ActorPPOTrainer` 对象，并传递必要的参数。
            - 如果需要，广播检查点到vLLM引擎。
            - 调用 `trainer.fit` 方法进行训练。
- `ppo_critic.py`
    - `class CriticPPOTrainer(PPOTrainer)`
        - `ppo_train`:
            - 创建数据加载器 `dataloader`
            - 遍历数据，执行训练步骤并进行全局归约。
            - 计算并返回训练状态的平均值
    - `class CriticModelRayActor(BasePPORole)`
        - **初始化模型**:
            - 使用 `DeepspeedStrategy` 设置分布式训练。
            - 初始化评论者模型 `get_llm_for_sequence_regression`。
            - 创建并设置 `tokenizer`（如果需要）。
            - 创建优化器 `critic_optim`。
            - 设置调度器 `critic_scheduler`。
            - 启用梯度检查点（如果需要）。
            - 准备模型、优化器和调度器。
            - 加载检查点（如果需要）。
            - 禁用 `use_wandb` 以避免重复日志记录。
        - **前向传播**:
            - 进行模型的前向传播，返回计算的值。
        - **追加经验**:
            - 将经验追加到经验回放缓冲区。
        - **训练模型**:
            - 清理CUDA缓存。
            - 调用 `trainer.ppo_train` 方法进行训练。
            - 清理经验回放缓冲区和CUDA缓存。
        - **清理缓存**:
            - 清理CUDA缓存。
        - **保存模型**:
            - 调用策略对象的 `save_model` 方法保存模型。
        - **保存检查点**:
            - 保存模型检查点。
- `vllm_worker_wrap.py`
    - `class WorkerWrap`
        - **init_process_group**:
            - 初始化一个用于模型权重更新的分布式进程组。
            - 使用 `init_process_group` 方法创建一个新的进程组，加入 `rank` 和 `world_size` 信息。
            - 打印初始化过程中的详细信息。
        - **update_weight**:
            - 从源 `rank 0` 广播模型权重到所有 vLLM 工作者。
            - 确保数据类型和模型配置的数据类型匹配。
            - 使用 PyTorch 的 `broadcast` 方法在分布式进程组中广播权重。
            - 加载新的权重到模型中。
        - `ppo_train`
            - 刷新经验数据。
            - 如果需要，远程训练评论者模型。
            - 训练演员模型并广播权重到vLLM引擎。
        - `training_step`
            - 调用 `training_step_actor` 方法进行训练。
        - `_broadcast_to_vllm`
            - 清理CUDA缓存。
            - 遍历模型参数并广播权重到所有vLLM引擎。
- `vllm_engine.py`
    - `class LLMRayActor`
        - **init**
            - `LLMRayActor` 类被定义为一个 Ray 的远程 Actor。
            - 初始化方法中根据 `tensor_parallel_size` 确定是否使用 GPU 执行器。
            - 根据 vLLM 版本和执行器类型设置相应的 Worker 包装器。
            - 初始化 vLLM 实例
        - **function**
            - `generate` 方法：调用 vLLM 的生成方法。
            - `init_process_group` 方法：初始化进程组，用于分布式训练或推理。
            - `update_weight` 方法：更新模型权重。
            - `stop_remote_worker_execution_loop` 方法：停止远程工作者的执行循环
    - `function`
        - `create_vllm_engines` 函数用于创建多个 vLLM 引擎实例：根据`tensor_parallel_size` 确定每个引擎使用的 GPU 数量和调度策略,使用 Ray 的 `placement_group` 创建资源绑定,初始化并返回 vLLM 引擎列表。