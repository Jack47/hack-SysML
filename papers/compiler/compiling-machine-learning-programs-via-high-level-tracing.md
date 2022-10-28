1. 在解决的是什么问题？给纯 Python 和 Numpy 的 ML 程序产生高性能加速器的代码
2. 为何成功，标志/准是什么？利用已有的 XLA 编译器基础设施来给 subroutines program 产生加速器最喜欢的代码，这些优化后的代码片段可以被任意的python代码调用和编排
3. 在前人基础上的关键创新是什么？结合了 Autograd、Numpy，支持"structured control flow"，可以在给复杂的机器学习代码生成加速代码的同时，维持高性能
4. 关键结果有哪些？易用性和性能都很好，支持多种硬件，还能扩展到多核上
5. 有哪些局限性？如何优化？静态shape?
6. 这个工作可能有什么深远的影响？

## 1 介绍

既要对研究员友好又要最大化使用算法的两个目标之间紧张是紧张的。比如动态语言Python编程很方便，但是太不受限制了而不能开启优化代码生成。同时高效硬件加速要求尽可能多的信息是静态的(DLVM, XLA)。

我们的解决方案基于经验上的观察：ML 负载经常是由大的，可加速的，pure-and-statically-composed (PSC)子函数组成，被动态逻辑编排。纯函数指执行函数没有副作用(?)。它是静态组合的，用的是一套‘primitive’functions的集合，可以被表示为一个由这些原语函数组成的静态数据依赖图(static data dependency graph)。
而提供的这些原函数是可加速的，比如，当它们组成 array-level numerical kernels和受限制的控制流(restricted control flow)，PSC routines 是加速的 prime candidates:他们详细描述了原始python程序里的片段，而其中所有未使用到的动态机制都被剥离（未被使用到？还是说不支持的，就还使用python？）

## TODO
The NumPy array: a structure for efficient numerical computation(2011)
## 问题
1. array 
