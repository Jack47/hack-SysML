modelmesh: Distributed Model Serving Framework

* high-scale
* high-density
* frequently-changing model use cases

能和已有的或者用户构建的 model servers 一起工作，给推理模型的运行时，扮演一个分布式的 LRU cache

这个 [pdf](https://github.com/kserve/modelmesh/files/8854091/modelmesh-jun2022.pdf) 里能看到支持的特性和设计的细节

为了完整的k8s纸上的部署和管理 ModelMesh 的集群和模型，可以参考 ModelMesh Serving  repo： 包含了一个分离的 controller 并提供了 K8S CRD 的管理，管理 serving runtime 和 inference services。有通用的模型repository存储的管理，可以和已有的 OSS model server 集成

## 介绍
* 成熟、通用的模型 serving 管理、路由层
* 给生产级别的高扩展，高密度和高频率改变的模型使用场景
* 主要给“模型当做数据”(什么特点？比较大？而且有很多版本？）的场景，而非“模型当做代码”
* 尤其在模型之间的使用变化（使用情况、频率？）很大的场景下
* 支持了很多 IBM Watson AI 云服务、app，包括助手、Discovery，NLP

## 背景
* 2015 年，IBM NLC 是第一个 Watson 云服务，提供了完整的端到端的训练和serving 用户提供的数据
* 原始的架构包括产生一个专门的容器来服务每个训练好的用户模型
* 问题
  - 增长到万级别的模型，无法很好地 scale
  - 免费的计划 - 很多很少使用或者被浪费了
  - 导致大量的内存代价 =》 硬件代价 =》 成本高
  
* 调度系统能力有限 =》严重地影响了性能和稳定性
* 所有服务的模型都有单点故障(每个只有一个 container）
* 大规模宕机后，启动、恢复的时间过长
* 在 k8s 之前，serverless 是普遍存在的

## Model-mesh
* 新方法
  - 每个 container 可以服务多个模型
  - 让冬眠(dormant)的模型可以被“paged out”，即席地加载
  - 代价比同时服务所有的模型要低几个数量级
  - 与 "serverless" 基础架构很类似（knative, cloud functions, ...)
  
* 哲学
 - 让规模化服务线上模型和特定的推理逻辑、技术解耦。隐藏了所有的（通用）考虑
 - 只做一次这些事情，做好

## 关键的设计目标
* 可扩容性
  - 不仅从管理的模型数量，而且从这些模型的使用容量上
  
* 性能
  - 最小化运行时请求的 latency
* 效率
  - 优化可用的计算资源
* 灵活性
  - 语言、ML-框架无关，可以使用自己的推理 API
  - 适配到不同的使用场景下，大小、性能的折衷等
* 简化集成，缩短展现业务价值的时间
  - 缩小服务提供者的要求、考虑的点
  - 应该是即插即用
* 运维简单
  - 最小化搭建和后续运维的需求、考虑
  - 配置和调优自动化
  
* 弹性(Resiliency)并且最小化“cache misses”的代价
  - 即考虑可用性有考虑对延迟的影响

## 架构
![](imgs/model-mesh-arch.png)

* Model-mesh 是单个的 docker 镜像，和“model 运行时”的 container运行在一个 Pod 里，部署并扩展，作为单个 self-contained 逻辑上的 k8s deployment，被当做一个标准的 K8S 服务来消费

## 数据模型
"Models" 有唯一的 id，被假设为无法被修改的

* 通常包括一个完整训练完之后的静态的产出物
* 包括一个训练号的 NN 的权重，或者其他用来预测或者推理的产出物。包括字典或者其他数据结构，来做前后处理
* 他们放置在共享的数据仓库里，可以拉取到内存里。模型之间不需要一致，尤其是消费的内存都不需要一样

"V-models" 是可修改的别名，指向单个的具体的模型：
* 可以被寻址
* 目标模型可以被动态更新
* 用来做线上模型的版本管理

## 已经可用的 ML 框架集成
* 可以和现有的模型服务一起工作，支持模型的多租户和动态重配置已经加载的模型 -- 自研的或者第三方的

* Built-in 适配器，可以和架子上的 models server 配合：
  - TF
  - Nvidia Triton ( TF 1&2, PyTorch, ONNX)
  - Seldon MLServer (包括 scikit-learn, XGBoost, LightGBM)
  - Intel OpenVino Model Server

## 贡献到 KServer 开源项目里：[modelmesh-serving](https://github.com/kserve/modelmesh-serving)
* ModelMesh Serving: 使用 k8s 原生的管理层来配套 model-mesh
* ServingRuntime CR 来指定每个模型的 server 配置
* 基于 Go 的controlner 来编排不同种类的 modelmesh 部署，都在同一个服务 endpoint 之后
* Models 可以被服务并被通过 Predicator 的 CR 来管理（会被 KServe 的已有的 InferenceService替代） - 映射到 mm 模型和 vmodels
* 给特定的 ServingRuntime 的 Pods 只会被有需要它的 Predictors 时才会被启动
* 提供通用的抽象的运行时无关的存储处理（用来获取模型）
* 提供内置的和已有的标准的 OSS 模型服务器集成后的方法，通过 injected adapter 来做
* 支持 KServe V2 REST API，通过可选的的注入的 proxy container

## KServer ModelMesh Serving Architecture
model-mesh 如何在KServe 里工作：

![](imgs/kserver-model-mesh-serving-arch.png)

## 针对已有的 Model Server 的适配器模式
* Puller 是被自动注入，然后作为 KServe 的一部分
* 否则，adapter 或者 model server 需要负责拉取模型数据
* ID 注入容许直接推理路径而无需关注传递给 adapter ？ 没懂，噢是约定了路径了，Puller 拉取到约定的路径

![](imgs/adapter-pattern.png)

注意上图里的管理面和数据面 API

注意：这些图覆盖的只有核心的 model-mesh container，没有包含如何被管理和暴露给 KServe 的

## APIs
Model-mesh 使用 gRPC 并基于三个逻辑上的服务 APIs 和两个逻辑上的服务边界（内部和外部）：
1. 内部模型管理 SPI(internal) # 指不需要干预的，是自动做的
   - 两个主要的方法：loadModel 和 unloadModel
   - 由内部的“模型运行时”容器实现 - 从存储里加载指定的模型到内存，准备好服务
2. 外部的模型管理 API (external) # 指跟业务相关的，被调用的接口。类似我们的 MR
   - 由 model-mesh 暴露的，提供方法来注册和取消注册新模型到平台里，创建、更新、删除 vmodels
3. 运行时推理 API (both）
   - 单个或者多个任意的 gRPC 服务定义，新的或者已有的
   - 包裹了推理的逻辑来调用已经加载好的指定id模型
   - 由内部的“model runtime”容器来实现，自动、透明地被外部的 model-mesh 服务暴露出去
## Internal SPI

## Internal SPI 的要求和保证
要求：
* 必须支持并发（多线程）的 API 请求 # 不再是强要求 -- 查看 “基于延迟的 autoscaling
* 必须能相对精确地测量加载的模型的大小 # 目的是尽量放更多的模型在一个卡上
* 加载的模型能一直可用，除非被显式调用 unloadModel 卸载（没有驱逐或者无须 cache management)
* 推理、调用 API 必须从 gRPC meta 数据参数（payload 无关的 header）里读取目标 modelId，是幂等（可重试）

保证：
* 不会有超过最大值的 loadModel 请求并发在途中
* 模型调用请求只会在已经加载了的模型上

## External API
是由 model-mesh grpc-service 来暴露的。所有的方法都是幂等的

* registerModel (modelId, modelInfo, loadNow, sync)
  - 返回当前结果
* unregisterModel(modelId)
  - removes/unregisters model from the service
* getStatus(modelId)
  - returns one of NOT_LOADED, LOADING, LOADED, LOADING_FAILED, NOT_FOUND
* ensureLoaded(modelId, sync, lastUsedTime)
  - ensure model is loaded, optionally with specified last-used timestamp; returns current status
* setVModel(...), deleteVModel(...), checkVModelStatus(...)

* Plus, 所有的推理 API 方法都是由配置好的 model server 容器暴露的
## 和消费者服务集成
* 调用的是 model-mesh 暴露出来的 API 方法
## model ID 抽取和注入

## 特性 - Cache 管理和 HA
* 集群里的实例都被当做一个内部的**分布式的 LRU cache**，而可用的 model server 容器填满了注册好的模型
* 运行时的请求可以到任意pod里，会被转发（按需）到有目标模型的容器里
* 如果目标模型没有被记载，会立马触发一个加载（在看起来最优的实例上），请求会被延迟直到加载完成
  - 自动做的 - 如果多个请求在同时到达，只有 exact intended number of copy loads 会被触发
  - 这样避免额外的 churn、load
## 特性 - Cache 管理和 HA
管理有**多少特定模型**的拷贝被加载，在哪里，什么时候，都是在框架里自动完成的。包含：
- 确保所有”最近“使用的模型至少有两个副本
- 扩展到超过，只有当确定的请求的load阈值超出了 - 一个模型在足够的load下可以在每个集群实例里有一份拷贝
- 缩小到大于2到2，从2到1是**基于启发式 load 和最近最后使用的时间来做的** -- 最忙的实例优先
- 主动**加载**最近没被用的模型，如果有空余空间或者替换最近没怎么被使用的加载的模型 # 为什么要替换？万一不划算

可视化典型的 cache 里的分布
![](./imgs/visual-representation-of-typical-cache-distribution.png)

* **global LRU** 是最近最少被使用的，又被集群里加载了的模型的最后被使用的时间
* 框架确保自从这个时间开始被使用的模型，都已经被加载了，可以立马使用
* 只有当集群满了的情况下，才有效果 -- 没满的情况下，所有注册的模型都可以有恰当的实例
* 这是跟踪一个模型 serving 集群的容量的关键指标，和 cache misses(无法立即被服务，只能等待模型加载完）的频率关联
* 它的值在随着时间变化，而且随着集群大小的变化，可以被调整 -- 是一个很合适的指标来做 auto-scaling deployment 大小，但是基于经验，变化很小，所以不是很需要
* 一些实现下，model 加载需要好久，无法接受在推理请求里进行等待，那就设置 pod count 到足够大，来加载所有的模型

## 特性 - 升级，迁移，集群扩缩
都是下面的步骤：
- 立马被删掉
- 本地加载的模型被 propagated (in parallel) 到其他实例，基于 balanced/optimal placement，基于最近使用情况来做优先级排序。全局的 LRU 位置和聚合后的copy数量也是被传播
- Queued 或者正在加载的模型被放弃
- 实例等待正在和最近被使用的模型加载完
- 从服务发现里反注册，持续服务到几秒后，等待 in-flight 的任务完成
- container 退出

## 特性 - 异构的集群

## 优化 - controlled，Prioritized loading
* 框架控制每个实例上并发加载的模型个数。loading的代价取决于 model server 实现，可以指定一个合适的大小
* 这个很重要：确保加载的活动不会要想到 latency 敏感的推理请求
* 模型在实例上的放置有最大的加载容量

## 运行时推理的弹性
* 一个 model-mesh 的部署其实形成了一个分布式的“service mesh”。从外部来看像其他集群的服务，但是内部请求是被路由的
* 每个容器级别的 CPU 资源分配，确保不会互相影响
* 一个请求因为链接原因失败，可以自动重试，放到下一个有此模型的实例上，或者当做 cache-miss

## 模型加载的弹性

## Rate-Tracking 和 Auto Scaling
* 实时的推理请求，在每个节点上，以**一个实例**和**每个模型粒度**做统计
  - 每10秒访问一次，只有在之前的时间窗口里的模型需要处理
  - 这些用来做后续的放置和扩容的决策
  - 最大负载、最高频使用的模型会被放在最少负载的实例和以此类推 -- 比如老的模型会被主动加载，来实际在 heavily loaded 实例上（有空闲的内存）
  - 这个策略被证明在维护**大集群里整个负载的均衡**方面是非常有效的
* Scal-up 一个模型，是用去中心化的方式进行的
  - 每个实例会对自己服务的 active 模型形成扩容的决策，基于动态可配置的**每个实例级别的 request-rate** 阈值
  - 由于推理请求实在模型的副本间均衡的，所以会让实例"共享“后续的请求，逐步就会让请求掉落到阈值以下

## Latency-based Auto Scaling
* New auto-scaling mode，当前是在 beta 状态
* 在一个 model-mesh 集群里全局开启
* 和现有的 request-rate-based auto-scaling 的不同：
  - 每个加载的模型汇报 **inflight requests 上的最大并发**
  - Model-mesh 会**按需排队**
  - 扩容是按需触发，目的是为了最小化排队时间
  - 显示的 request load 阈值参数不再需要了
  - 尤其适合：
    - 模型的推理逻辑不是多线程的
    - 每个请求的推理代价和耗时较高或者可变的

## 实现细节 - 模型加载数据
* 跟请求速率一样，聚合后的模型加载时间统计是持续跟踪的（平均和方差）

* 持续变化的状态（模型使用次数等）是用懒更新的方式，精确的cache ordering，每个模型的使用速率，模型大小信息都是仅仅保存在本地，通过内部的 RPCs 来发送的

## 实现细节 - 后台任务
* Janitor
- 每个实例里，每隔6分钟运行一次
- cross-references and reconciles 本地的cache 和 model registry
- Scales-down model copies as/when appropriate

* Rate Tracker
- 每10s在每个实例里运行一次
- 处理自上次以来运行的模型，有需要就触发扩容

* Reaper
- 在选出的单个 leader 上，每7分钟执行一次
- 跟踪”缺失“的，在MR里的实例引用，清理孤儿注册
- 如果集群里有空闲的空间，就触发主动的模型加载，或者其他最近使用，但是没有被加载的模型

## 实现细节 - etcd registry
* etcd 里有三个表 - models, vmodels, instances
* 每个实例关注这些表，而且在本地内存里存放镜像
* 几乎所有场景下的读取都是从本地内存读
* 条目都是 json 格式，但是大小是保持在最小（后续再考虑切换到 protobuf）

* **models**表有一个条目，给每个注册的模型：
- 每个实例内部的本地 cache 里的条目是和 模型的 logical weighted LRU cache 分离的
- 条目包含：最小的 model server 用来 local/load 模型的数据；模型被加载、加载中、失败的 instance ids；refcount 和一些在loading、unloading决策里用到的 timestamp

## 实现细节 - etcd Registry
instance 表里有每个运行的实例的信息，只会被这个实例创建、更新，而且会在实例关闭时删除：
- 这些条目绑定在跟实例关联的 session lease 上，所以会在实例死亡后很快就自动删除掉
- 包含这个实例的高级别的信息，当这些信息修改时更新，但是会限制在每秒更新一次上；实例里 LRU 的模型时间，模型个数，容量，使用，正在做的加载，当前 request load (最近几分钟里 rpm 的均值），其他一些数据
- 这些值用来在当一个模型的副本需要被放置时，做决策

## 实现细节 -etcd registry

每个 etcd 里模型 记录的生命周期
- 注册时产生，反注册时删除（实例独立的）
- 有对应的实例的id，用来进行路由
- 在加载模型前，实例把自己的 id 添加到模型的记录里
- 每个实例里有个定期任务(Janitor)，会 review 它加载的模型，决定unload哪个。当 unload 决策作出后，首先会从 model record 里删掉这个实例id
- 在 leader 实例里有个全局的定期任务(reaper)，从模型的记录里清理那些没有优雅退出的实例id

## 实现细节 -etcd registry
## 快速开始

1. 在这个 modle-runtime.proto gRPC 服务接口里，Wrap  模型的加载和调用逻辑
   * runtimeStatus()
   * loadModel() - 从背后的存储，加载指定的模型到内存里，阻塞的
   * unloadModel() - 卸载掉
   * 推理的接口，可以有多个，应该是幂等的接口。可以参考 predictor.proto 来作为一个简单的例子
   
2. 构建一个镜像，在 8085 或者 unix domain socket 上暴露服务

3. 扩展
4. 部署服务
   * registerModel() 和 unregisterModel() 来注册或者删除掉集群里的模型
   
   
## 问题
1. loadModel() 背后的 cache 机制看看？比如此时请求是怎么被延时处理的？
2. registerModel() 时怎么处理的存储的问题？
3. ServingRuntime CR 来指定每个模型的 server 配置:  这个配置里有什么？
4. ModelMesh 这个 sidecar 怎么和其他人配合的？可能要看极客时间的文章
5. per model 或者 per cluster 级别的 metrics 数据（Rate Tracking，Janitor），是怎么做到的？
