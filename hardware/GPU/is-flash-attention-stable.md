1. 在解决的是什么问题？想看看 FA 给训练带来的影响
2. 为何成功，标志/准是什么？定量分析了 fa 对激活值的影响，同时也分析了它对模型权重的影响
3. 在前人基础上的关键创新是什么？用了 Wasserstein Distance 这个指标，可以衡量两个 tensor 之间的差异分布
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

论文里面最终想要讨论的是（不同精度、不同初始值、FA）这些因素，谁对权重的影响更大

单纯FA的计算误差确实会更大

上面的图里面Flash vs Baseline的魏氏距离也不是0。FA虽然有误差，但是相比随机初始值和低精度，还是要强


论文里提到了几篇从算法角度考虑训练不稳定问题的：
1. A loss curvature perspective on training instability in deep learning 2021
2. A theory on adam instability in large-scale machine learning (2023)
3. Surprising instabilities in training deep networks and a theorical analysis (2022)
