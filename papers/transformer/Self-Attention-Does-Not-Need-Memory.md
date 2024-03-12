1. 在解决的是什么问题？ 让 O(n^2) 显存降下来，但是计算复杂度没变
2. 为何成功，标志/准是什么？这种 attention 计算下，attention 显存跟序列长度关系是 O(1)，即常数的关系，而 self-attention 需要 O(logn) 显存。复杂度没变，依然 O(n^2)。在 TPU 上实现了 O(sqrt(n)) 复杂度显存，数值稳定，只有标准 attention 
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

在 xformer 里有对应的实现：`memory_efficient_attention`，它的思路并不会加速，因为它主要目的是降低对显存的需求，而非加速。

上面为了得到O(sqrt(n)) 的显存复杂度，keys 和 values 也需要是 sqrt(n) 的 chunk size
