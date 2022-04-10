
PaLM: Scaling Language Modeling with Pathways

a 540-billion parameter, dense decoder-only Transformer model trained with the Pathways system. PaLM 证明了首次大规模使用 Pathways 系统扩展到 6144 个 芯片上。是目前位置
最大的 TPU 系统配置。它是通过数据并行泡在Pod level上，跨越了两个 Cloud TPU v4 Pods。而每个 Pod 内部使用标准的数据和模型并行。说明每个Pod（islands）上，有 3048 个芯片

PaLM 的训练效率达到了 57.8% 的硬件 FLOPS 利用率。这个是 LLMs(Language Large Mode)首次在如何大规模上应用。这是因为并行策略的组合以及 Transformer 里容许 attention 和 feedforward 层可以
并行计算，增加了 TPU 编译优化的速度。
