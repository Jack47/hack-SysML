1. 在解决的是什么问题？在此之前，翻译领域的模型都是复杂的 RNN 或 CNN，里面包含 encoder 和 decoder。问题是并行化程度不高，训练所需的时间长
2. 为何成功，标志/准是什么？并发度更高，训练所需的时间更短
3. 在前人基础上的关键创新是什么？提出了注意力机制，使用了 scaled dot-producted attention, parameter-free position representation.
4. 关键结果有哪些？泛化效果好
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 问题
1. Q K V 的矩阵乘里，Q 和 V 分别是什么？
2. 是不是里面有 encoder、decoder，positional embedding 的介绍？
3. attention mechanism 到底是啥？
