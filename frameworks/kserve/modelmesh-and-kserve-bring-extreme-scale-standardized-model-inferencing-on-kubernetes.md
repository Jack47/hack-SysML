
![](.imgs/model-mesh-and-kserve-arch.png)

## Core Components
[ModelMesh Serving](https://github.com/kserve/modelmesh-serving): 是 model serving 的 controller (Go)

ModelMesh：containers，用来编排模型的放置和路由策略 （居然是 java 写的）

## Runtime adapters
在每个 model-serving pod 里运行，扮演一个在 ModelMesh 和第三方 model-server 之间的中间角色，同时有 puller 逻辑来负责从存储里拉取模型

## Model-serving runtimes
支持 triton-inference-server
