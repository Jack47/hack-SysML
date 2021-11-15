## 1. 简化，简化，简化
当求解涉及向量的导数时，大部分困难是由于一次做了太多的事情：多个部分一起求解，在求和表达式上求解，应用链式法则

### 1.1 对每个部分分解为单个标量的计算
即写出 yij 与 xij 的关系，然后针对标量对标量进行求导。矩阵的求导，可以分解为每个单独标量元素的表达式

如果 y 里有 C 个元素，x 中有 D 个元素，那 d(y)/d(x) 里，有 C*D 次运算


如果 y 里有 C*D 个元素，x 中有 D*E 个元素，那 d(y)/d(x) 里，有 C*D * D*E 次运算

### 1.2 去除求和符号
求导之前，最好先去掉求和/积符号，把它们展开来，之后再关注y中的一个标量和x中的一个标量来求导

通过手写发现：

y = W*x, d(y)/d(x) = W

[Vector, Matrix, and Tensor Derivatives(Erik Learned-Miller)](http://cs231n.stanford.edu/vecDerivs.pdf) [My Notes](./courses/cs231n/vector-derivatives.md)
