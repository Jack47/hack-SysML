1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 1 Introduction
对于需要使用 pipeline并行技术来让显存能放得下
DeepSpeed Inference，主要给 Transformer model 推理使用的，主要包含两个组件：

1. DeepSpeed Transformer: 是一个只给 GPU 的解决方案，目的是减小延迟的情况下，最大化Dense和Sparse Transformer的通吐。它
能给各种不同大小的 transformer上都达到 SOTA，单卡或者扩展到百卡的推理
2. ZeRO-Inference: 异构的 GPU+CPU+NVMe的方案来解决大模型推理时的内存不够问题

论文的主要贡献:
* 单个GPU上的 transformer kernel，利用显存带宽集中融合的调度和 GeMM kernels 来达到最小延迟和最大吞吐
* 一个多GPU上稠密的 transformer 推理系统，能 tensor 并行来降低延迟，流水并行和显存优化来提高吞吐
* 大规模GPU上稀疏模型的推理系统，结合了：i) expert, data, tensor 并行 ii)创新的通信优化 iii) 稀疏kernel优化，可以扩展到稀疏推理
* ZeRO 推理
## II Background and Related Work

## III Inference optimized transformer kernels
讨论挑战、设计和优化，可以让tansformer kernels能达到在大小batch都高效推理

### A Inference Challenges on Different Batch Sizes

A. Inference Challenges on Different Batch Sizes
小batch下的性能主要受限制于读取模型权重时的显存带宽。主要由三大挑战：

1. 由于 transformer里很多kernel(十几），导致小 batch 下 kernel 发射的开销太大
2. 每个kernel发射后写回数据到 global memory，这个会在下次kernel调用后再读取，这样数据就来回在主存和GPU 核心之间传输，额外开销
3. cuBLAS 和 CUTBLASS GeMM 库对**极端小(1?)的 batchsize 上表现不好**，无法达到好的显存带宽利用率

提出了2个技术来解决这些问题：

1. Deep-Fusion 来减少 kernel 调用和数据移动的开销：把多个 kernel 融合为一个
2. 定制的 GeMM kernel来提高bs很小情况下的显存带宽利用率，而且容许它可以和Deep-Fusion一起融合

B. Deep-Fusion
监管 operator fusion是常见技术来消除 kernel launch 和数据移动的开销，但是主要限制在element-wise op上。对比而言，transformer里有很多
op比如数据布局转换，reduction(softmax) 和 GeMM 操作，这让跨 blocks 之间有了数据依赖，导致很难 fused。为在GPU上，如果一个thread-block生产的数据
被另外一个消费，那就需要通过global memory同步，然后触发新的kernel
为了避免全局数据同步，DeepFusion 把计算空间沿着迭代的空间进行分片，这样不涉及跨分片的数据依赖，能够在不同块之间并行执行。这样计算空间里的维度就不包含数据依赖，不会被分片，被同一个线程块处理。
经过这样分片，如果每个tile里的第二个算子只依赖于第一个算子的输出，那两个算子可以通过DeepFusion被融合到一起。通过在分块粒度执行融合，DF 不仅可以融合 element-wise 的算子，还能执行 reductions，transpose 和 GeMM，只要没有跨分块的依赖。比如，所有layer-norm 里的 micro-op 都可以沿着token维度进行分块，而规约维度是在同一个分块里处理的。这样让同一个layernorm里的 micro-op 可以融合到单个的kernel，尽管有多个规约op。而且，被每个分块产出的数据可以在寄存器或者共享内存里，这样让多个op里的数据重用很方便，而避免了从global memory进行数据传输的开销。

C. SBI-GeMM: Custom GeMM for Small Batch Size
我们定制的 GeMM 实现是可以与 DeepFusion融合，而且达到最大显存带宽的。它有三部分组成：tiling strategries, cooperative-group 规约，为了更好的显存带宽利用率而使用的数据布局转换

1. Tiling Strategies: 图1描绘了我们给一个 skinny matrix multiplication 的 GeMM scheduling。受限沿着计算的输出维度分块。这样可以实现一个GeMM来使用一个kernel来让规约也能一起做。对于更小的模型，如果没发创建足够的并行分块来达到更高效的内存带宽利用，我们也会沿着输入维度进行分块，为了能够跨 tile 来规约而实现两个kernel
2. Cooperative-GroupReduciton：之前提到的分块策略下，在每个thread block里的warp，负责产出一部分规约的结果，最终的规约需要在当前thread block里的所有warps 上做。通常实现是在共享内存里基于 binary tree 来做，这需要多个 warp-level 的同步，因此有性能瓶颈。为了避免这个，我们执行了在共享内存里单个数据layout上的转换，让同一个输出上的部分结果在内存里是连续的，可以被单个warp使用 cooperative-group 来直接在寄存器里规约(见图1(a))。最后，每个warp里的第一个线程负责拿着最后的结果，写入到shared memory里。这样共享内存里的结果也是连续的，可以直接合并写入到全局显存里
3. 利用完整的 Cache-line：在GPU架构下，每个 L1 cache-line 是128字节，然而一个线程里对FP16或者INT8元素的合并访问并不能消费整个的cache-line。每个线程里，沿着输出维度读取多个元素可以解决这个问题，减少了并发分块的数量，这个对显存带宽性能很大。因此，我们的方案是把 weight matrix 在初始化时进行专职，让M行在内存里都是连续的，让每个线程可以沿着输入维度读取M元素(图1(b))。对fp16设置M为2 ,而 int8 是4。
