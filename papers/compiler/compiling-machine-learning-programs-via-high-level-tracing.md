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

JAX 系统是即系编译(Just-in-time)的编译器，通过高级别tracing以及 XLA 编译器基础设施来产生 PSC 子程序上的代码。之所以说JAX 里的 tracing 是高级别的，主要是以下两点：

1. 实现在源码语言纸上的用户级别的上，而不是作为源码里的一部分
2. trace的原语不是在VM 级别基础数据上的操作，而是在 lib 级别的数值函数，在数组级别的数据上，比如 mul，conv，沿某个维度的 reduction，elementwise 操作，多维度 indexing 和 slicing

JAX tracing 库是和 Autograd 内部使用的 tracing lib一样的，这个库设计用于 self-closure, recognizes its own operations as primitives. JAX 也有 Numpy 的数值函数。因此它给用类似 Numpy 风格的python函数产生代码，覆盖了任意阶的 forward 和 reverse-mode 自动微分。在后端，JAX 使用 XLA 来做array-level的程序加速和代码生成。与其他专注于提供有限集合的手写代码，特定的数值kernel的系统不同，JAX 提供所有 XLA 上支持的目标架构：通过 trace-compiling PSC 子程序，JAX 自动从已有的里分阶段出来新的kernel(?)

JAX 缩写指的是：Just after execution，因为为了编译一个函数，我们受限要在python里监控它的执行

## 2 系统设计
JAX 的设计受 ML 负载里通常被 PSC 子程序支配的启发。因此 JAX 只通过要求用户注解 PSC 入口来 trace 信息。即一个Python函数，它满足 PSC 的假设。这个设计利用了这些函数很容易在 ML 代码里找出来的特性，这样研究员很容易就能用 JAX 的 `jit_ps` 来注解出他们。JAX 有 trace 的cache，因此只有新遇到的 array element 类型，维度或者 tuple 会触发新的编译

JAX 会把 python代码 trace 出一张执行图，里面是原语的函数，有着静态数据依赖。现有的原语不仅由 array-level 的数值函数组成，比如 Numpy 和其他 conv、windowed reductions 之外，还包括有限的控制流函数，比如 `while_loop` 和 `cond`。这些控制流原语能让用户 stage control flow 到 compiled computation 里去，还保留了 PSC 的特性。最后，JAX 包含了一些函数式分布式编程的原语，比如 `iterated_map_reduce`。这套原语集合是用 Python 定义的，而且**可扩展**。新的原语只需要使用一个转换规则来注解，可以构建对应的 XLA 计算

为了产生代码，JAX 把 trace 转换为 XLA HLO，它是一种中间语言(IR)，可以建模出高级别的可加速的array-level 数值程序。广泛来说，JAX可以被看做是一个把 XLA 编程模型搬到 Python 里去的系统，让它可以用加速的子程序，同时允许动态编排(while，if)。

```
def xla_add(xla_builder, xla_args, np_x, np_y):
    return xla_builder.Add(xla_args[0], xla_args[1])

def xla_sinh(xla_builder, xla_args, np_x):
    b, xla_x = xla_builder, xla_args[0]
    return b.Div(b.Sub(b.Exp(xla_x), b.Exp(b.Neg(xla_x))), b.Const(2))

def xla_while(xla_builder, xla_args, cond_fun, body_fun, init_val):
    xla_cond = trace_computation(cond_fun, args=(init_val,))
    xla_body = trace_computation(body_fun, args=(init_val,))
    return xla_builder.While(xla_cond, xla_body, xla_args[-1])

jax.register_translation_rule(numpy.add, xla_add) # 意思是它自己也是这样实现的？注册了用户使用的函数和XLA之间的对应关系
jax.register_translation_rule(numpy.sinh, xla_sinh)
jax.register_translation_rule(while_loop, xla_while)
```
最后，JAX 完全兼容 Autograd(意思是可以使用这个包)。举例：
```
import autograd.numpy as np # 居然不是用的自己的 numpy 的包，那怎么做到识别出里面的函数用了哪些？
from autograd import grad 
from jax import jit_ps

def predict(params, inputs):
    for W, b in params
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs
    
def loss(params, inputs, targets):
    preds = predict(params, inputs)
    return np.sum(preds-targets)**2
        
```
## TODO
The NumPy array: a structure for efficient numerical computation(2011)

Parallelizing Julia with a Non-Invasive DSL(2017)

Casette.jil 2017

HPAT: high performance analytics with scripting ease-of-use. 2016

## 问题
1. 它是否支持 dynamic shape？ dynamic control flow？目测是不支持 dynamic control flow 的
2 

## 启发
1. 类似 XLA 里可以打印 HLO，以及 fusion 之后的 HLO 一样，是不是，是不是 TD 里也可以用这个做可视化？
