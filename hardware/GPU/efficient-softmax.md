## Softmax 函数
它能将一个含任意实数的 K 维向量 z “压缩”到另一个 K 维实向量中，使 **每一个元素的范围都在 (0, 1)** 之间，并且 **所有元素的和为1**:

```
feth(z)j = e^zj/sum(e^zk) for j = 1,...,K
```
Softmax 函数实际上是有限项离散概率分布的梯度对数归一化

```
z = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
np.exp(z)/sum(np.exp(z))
```
作用：对向量进行归一化，凸显其中最大的值并抑制远低于最大值的其他分量

## 问题
### 1. CUDA的实现里，为何需要计算 ReduceMax？lightseq 的论文里提到了:
为了避免混合精度训练过程中出现溢出，可以将输入进行等比例平移，这样结果不变：

np.exp(z-max)/sum(np.exp(z-max))

这样需要三步：

1. 找到输入中最大的元素，记为x'
2. 给每个元素扣除 x'，这样保证指数不会溢出，然后计算分母：Z = sum(np.exp(x-x'))
3. 计算 Softmax：yi = exp(xi-x')/Z

上述1和2都是规约的操作：求全局最大值，求和


## 参考资料
1. [如何实现一个高效的Softmax CUDA kernel？——OneFlow 性能优化分享](https://zhuanlan.zhihu.com/p/341059988)
