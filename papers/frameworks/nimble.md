1. 在解决的是什么问题？现有 DL 框架调度开销很大，并行度不高，没有充分利用 gpu 并发能力
2. 为何成功，标志/准是什么？ 比 PyTorch、 TensorFlow 速度快
3. 在前人基础上的关键创新是什么？ 将调度开销提前做了，然后通过分析 torchscript 里的计算图，进行最大并行度、最小同步点的并行化
4. 关键结果有哪些？简单、整洁，效率比其他要高
5. 有哪些局限性？如何优化？只能是静态图，静态输入
6. 这个工作可能有什么深远的影响？

## Stream Assignment Algorithm
两个目标：

1. Maximum logical concurrency
2. Minimum number of syuunchronizations
## 启发
1. static graph 里，确实调度的开销可以从训练的主循环里搞出来
2. 
## 问题
1. 在咋们的场景下，scheduling overhead(申请内存、) 是多少？
2. 如何与 checkpoint 配套？
