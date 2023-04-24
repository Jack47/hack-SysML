modelmesh: Distributed Model Serving Framework

* high-scale
* high-density
* frequently-changing model use cases

能和已有的或者用户构建的 model servers 一起工作，给推理模型的运行时，扮演一个分布式的 LRU cache

这个 [pdf](https://github.com/kserve/modelmesh/files/8854091/modelmesh-jun2022.pdf) 里能看到支持的特性和设计的细节

为了完整的k8s纸上的部署和管理 ModelMesh 的集群和模型，可以参考 ModelMesh Serving  repo： 包含了一个分离的 controller 并提供了 K8S CRD 的管理，管理 serving runtime 和 inference services。有通用的模型repository存储的管理，可以和已有的 OSS model server 集成

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
1. loadModel() 背后的 cache 机制看看？
2. registerModel() 时怎么处理的存储的问题？


