1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

本文里，发布了 Llama 2，是一堆预训练和精调过的大语言模型，从 7B 到 70B 参数。我们精调了 LLM，叫 Llama 2-Chat，是为对话场景优化的。我们的模型在绝大部分测试的 benchmarks 上都效果很好，而可以作为闭源模型的替代品。提供了详细的描述：如何精调，做安全提升。

## Model Details:
两类模型：

1. Pretrained Models：答案是 prompt 的自然补全
2. Fine-tuned Chat Models: 给对话应用使用的。

使用了 supervised fine-tuning (SFT) 和 有人类反馈的 RL 来对齐到对人类有帮助和安全

Content Length 4K, Tokens: 2T, 从一月训练到了7月。在遵守 Llama2 社区协议情况下，可以精调 Llama 2 模型到其他语言上

## 训练数据
在 2T 公开来源收集的 tokens 上做预训练。精调的数据也是从公开的指令数据里来，有一百万人类标注的例子。预训练或者精调数据集里没有 meta 的用户数据

SFT 里，几万条就已经足够了

