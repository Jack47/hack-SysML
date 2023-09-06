## Prompts

1.3 --> 每个单词平均对应的 token 数量

即750的文字，大概是1000个 tokens

## 价格

~50：1-- GPT-4 相比 GPT3.5-turbo 的价格

5:1 -- 生成文本时，使用 GPT-3.5-Turbo 和 OpenAI embedding 的对比

10：1 -- OpenAI Embedding 和自己搭建的对比

## 训练和 Fine Tuning
一百万刀：训练一个 130 亿参数的模型，在 1.4T tokens 上

LLaMa 13B 里，使用了 2048 块 80G 的 A100，花费了 21 天才训好

## GPU Memory
V100： 16GB，A10：24GB，A100：40/80GB. A10 比 A100 便宜一大半

2x 参数量：通常 LLM serving 所需的显存:

比如 7B，需要 14G。比如 llama.cpp 可以把 13B 跑在 6GB 的 GPU 上

~1GB：通常 embedding model 所需的 GPU 显存

`> 10x`：batching LLM 请求达到的吞吐提升：

~1 MB：13B 参数模型上输出一个 token 所需的 GPU 显存。即需要 512 tokens（大概380个字符），那就大概需要 512 MB
