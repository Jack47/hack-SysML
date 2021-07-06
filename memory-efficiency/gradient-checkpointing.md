1. 在解决的是什么问题？ 优化显存占用，能够训练更大的模型，更大的 batch size
2. 为何成功，标志/准是什么？ 
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？ 内存占用是O(sqrt(n))，额外带来20%～30%的计算开销
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？


[Visual Gifs to show gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing)

在图里，被选为 checkpoint 的节点，需要在forward 及之后都保持在内存里，而剩余的节点最多只重计算一次。这样重计算的节点数量在 sqrt(n) 级别。对于非 transformer 的网络结构，需要用户手工指定 checkpoints 的位置。

有几种用法来指定 checkpoint:

1. 手工指定：我们在定义 model 时，指定。
2. memory： 使用启发式方法，自动选择一系列节点来 checkpoint。比如把图分成两部分的连接点。
3. speed：最大化运行速度，通过 checkpoint 那些重计算比较慢的节点，比如卷积和矩阵乘


