Expert Parallelism 是另外一种模型并行，它主要是针对 experts 参数的。在 ep 下，不同的 experts 被放置在不同的设备上，并行执行。那他们之间就得靠 all-to-all 来通信
