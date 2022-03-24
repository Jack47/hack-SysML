1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

Reviewer 1 提到了：chunk based 这种优化在 HPC 领域很流行，作者实现了自己的硬件，但是没详细介绍。然后只在 CNN(Resnet) 的分类上做了测试。标题应该叫FP8结合的混合精度训练，因为并不是所有都fp8，而是在有限的卷积/矩阵乘这里用 fp8
## 参考材料

1. [NIPS Open Review](https://proceedings.neurips.cc/paper/2018/file/335d3d1cd7ef05ec77714a215134914c-Reviews.html)
