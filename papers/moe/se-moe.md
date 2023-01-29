1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 摘要
大量实现表明，SE-MoE 可以训练一个 统一 Feature 优化(Unified Feature Optimization(UFO)) 的模型，使用 Sparsely-Gated Mixture-of-Experts 模型，有120亿参数，8天里用48个 A100 GPU卡就训好了。比当前 DeepSpeed 的训练吞吐量要高 33% (tokens per second)，推理通吐量高13%。而且在不均衡的 MoE 任务里，比如 UFO,SE-MoE 可达到 64% 的吞吐量，而降低18% 的显存。

## 4

### 4.2 Resource-Aware Communication
在 MoE 模型训练和推理时，expert 并行的模式下，需要大量的 GPU 卡之间的 All2All 通信。
