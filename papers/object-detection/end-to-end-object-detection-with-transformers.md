1. 在解决的是什么问题？与传统 cv 里的不同，检测里平常又复杂的手工的流水线，而 DETR 把检测当作一个直接的集合预测问题(direct set prediction problem). 包含一个基于集合的 global loss，通过二分匹配的唯一预测(unique predictions via bipartite matching)和transformer 的 encoder-decoder 架构来做。
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

与传统 cv 里的不同，检测里平常又复杂的手工的流水线，而 DETR 把检测当作一个直接的集合预测问题(direct set prediction problem). 包含一个基于集合的 global loss，通过二分匹配的唯一预测(unique predictions via bipartite matching)和transformer 的 encoder-decoder 架构来做。给定一个固定的学到的对象query小集合,
DETR 推理出目标和全局图片上下文之间的关系来直接并行输出最后的预测集合。由于这个并行特性，DETR 非常快而且高效

![](imgs/detr-arch.png)

[Detection TRansformers github](https://github.com/facebookresearch/detr)
