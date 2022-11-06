1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？抽象了一套创新的用于 tensorization 计算的原语，可以作为未来开发 tensorization 感知自动优化调度的基础 。
3. 在前人基础上的关键创新是什么？ 1. a novel abstraction for tensorized programs. 2. 构建的转换原语来产生丰富的tensorized 程序优化空间，带正确性校验 3. 设计实现了 tensorization 感知的自动调度。
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 6 相关工作
TensorIR 专注于自动化  tensorization 处理来产生高效的多平台的代码，无须人工干预
## 7 总结
关键抽象：block，可以封装tensorized 计算，提供高效程序优化的转化原语。构建了一个自动调度的算法来执行 tensorization，可以和其他优化联合，产生高效程序。

