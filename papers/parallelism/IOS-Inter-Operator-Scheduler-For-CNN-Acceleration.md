1. 在解决的是什么问题？新的网络结构在新的GPU上利用率不高的问题
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？使用 dynamic programming 来解决问题，这样可以处理巨大的调度解空间，因为common sub-schedules among different schedules.
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

https://github.com/mit-han-lab/inter-operator-scheduler

多分支的CNNs 使用更多小的Conv，放大了现在新的GPU上资源用不满的情况

有三种方法：

1. Intra-operator parallelism

op 内的并行主要由GPU和框架提供

2. Inter-operator scheduling

3. Graph transformation
算子融合能提高更大并发（一个更大算子而非两个小的、顺序的算子）并减少对GPU显存的访问次数。比如 MetaFlow、TASO

