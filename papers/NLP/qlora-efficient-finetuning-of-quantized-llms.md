1. 在解决的是什么问题？可以在一个 48G 的 GPU 上微调一个 65B 参数的模型，同时保留了 16-bit 的微调任务性能。
2. 为何成功，标志/准是什么？只需要在单个 GPU 上 finetuning 24小时，就可以达到 ChatGPT 99.3% 的性能。
3. 在前人基础上的关键创新是什么？有4个
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

a: 4-bit NormalFloat(NF4), a new data type that is information theoretically optimal for normally distributed weights （噢是4bit 的正态分布后的权重？有什么前提要求吗？
b: double quantization 来减少平均的显存占用，方法是量化 quantization constants
c：paged optimizers 来管理峰值内存

在超过 1000 个模型上使用了 QLoRA 来微调，提供了 8 个指令数据集上的详细分析。而且发现 GPT-4 的评估是便宜而且比人类评估更合理的替代品。且我们发现当前的 chatbot benchmarks 是无能完全可靠来准确评估 chatbot 的性能的。


代码：https://github.com/artidoro/qlora

