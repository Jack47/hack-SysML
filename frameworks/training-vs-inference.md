## 问题：
0.self attention 和 cross attention 之间有什么不同？q 的来源不同，我们现在 llm 里用的都是 self
1.训练和推理的时候，计算 attention 的地方有什么不同？输入的长度不同
训练时是直接根据 input，计算出 q、k、v ？
2.q 是怎么计算出来的？推理时只跟上一个 token 有关

## 推理
kv cache

decode 和 prefill 阶段的差异：

prefill 时，attention 部分的输入是一整个 prompt，所以此时 q、k、v的大小都是 (b,s,h) （跟训练很像？）
而 decode 时，attention 部分的输入是上一次生成的 token，此时计算出的 q 是(b,1,h), 而k、v 是用 (b,1,h)拼接之前的 kv cache，组装成 (b,1+s,h) 大小，最终输出是 (b,1,h)

主要是为了解答 yaoxi 问的问题，kv cache 部分有什么不同：目前看到 vllm 里直接存储的是常见的那种方式，并没有存储 compressed ，所以利用不起来 MLA 的特性
