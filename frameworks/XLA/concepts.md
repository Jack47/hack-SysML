## XLA 支持的前端有哪些？
Tensorflow, Julia, Jax, PyTorch

## XLA 支持的后端硬件有哪些？
TPU，GPU，CPU

## HLO 是什么？
HLO(IR) 指 High Level Operator，高级运算。HLO可以简单理解为编译器IR

XLA里统一的图抽象的IR：XLA HLO，然后可以编译为目标机器上的可执行程序

XLA 里的OP集合比前端框架里的（比如TF)要小很多，只有80多个

## XLA 如何编译程序？

可以看到分为两个阶段：

1. 目标设备无关的优化
2. 目标设备有关的优化

![](imgs/how-does-xla-compiles.png)
