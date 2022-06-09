1. 在解决的是什么问题？真实场景下部署大模型遇到的算力瓶颈，搞了这个 BMInf，主要是大模型推理和调优(tuning)
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？ a.算法：模型量化和高效模型推理和调优里的参数高效调优(parameter-efficient tuning for efficient model inference and tuning). b. 实现层面：使用 cpu offloading, activation checkpoint 和 CPU-GPU优化调度
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 1 简介
一些最近的高效参数调优方法(Hu 2021， Ding et al. 2022)已经达到了和finetuning 所有模型权重一样的结果。我们提供了一个统一接口，来冻结大模型里所有权重，
只给额外的模型计算梯度，这样来支持大模型处理具体的不同任务，不同的高效参数的调优方法

## 问题：
1. 里面的 tunning 指？ finetune，指模型已经预训练好了，现在需要在下游具体场景上 finetune 一下
