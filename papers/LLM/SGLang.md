SGLang: Efficient Execution of Structured Language Model Programs

使用了一些创新的优化，比如 RadixAttention 来让 KV cache 可以复用，利用 compressed finite state machines 来加速结构化decoding 输出。

LM 程序里有两种特性：

1. LM 程序通常包含多个 LLM 调用，中间穿插了控制流。这是完成复杂任务，提高整体质量所需
2. LM 程序接收结构化输入，产出结构化的输出。用来把 LM 程序进行组合，集成 LM 程序到已有的软件系统管理去

在运行时，我们提出几个创新的优化方法：
1. Radix Attention： 让多次生成调用之间的 KV cache 自动复用。已有的推理引擎里，一个请求的 KV cache，会在处理完之后丢掉。组织了 KV cache 在多个生成调用之间共享。相反，我们的系统给所有的请求的 KV cache 用 LRU cache 的方式组织到一个 radix tree 里。这样把 KV cache 当作传统的 cache 来使用，使用 radix tree 来做高效的匹配、插入、删除。让 runtime 可以高效地以 cache-aware scheduling policy。

radix tree 时 trie (前缀树) 的变种，是空间更加高效的替代品。和传统的树不同，radix tree 的叶子可以被标记为不仅仅是单个元素，还可以有变长的元素。这样就更加高效了。paged 布局，每一页等价于一个 token。RadixAttention 和已有的技术比如 continuous batching、paged attention 兼容。对于多模态模型，可以扩展为处理 image tokens。


2. Compressed Finite State Machines： 让结构化输出时可以加速处理。不再是一个token的生成，而是一批token的生成

问题：compressed finite state machines 还是一个概率模型吗？还是说把有限状态自动机结合到模型里，当出现某个预测，就代表进入了状态机里。
