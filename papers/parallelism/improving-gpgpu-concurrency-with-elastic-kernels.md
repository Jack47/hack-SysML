1. 在解决的是什么问题？GPU上有不少（比如30%）资源没利用起来的问题
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？Elastic kernels, 能控制一个 kernel 的 grid 和 block的数量，进而控制资源分配量
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

主要贡献：

1. 识别出 CUDA 执行模型里，在有长时的kernel和显存转移存在的场景下的障碍。提出分时复用的

## 4. 实现
本来是按照 grid和threadblock 来分配资源的，但是 elastic kernels可以通过把原来的kernels通过源码转换为 elastic kernels， 是软件实现的。这样就可以做一些资源感知的调度

限制：对于有 shared memory 使用以及 synchronization 指令的代码，不适用。

## 问题
false serialization 是什么意思？
