1. 在解决的是什么问题？self-attention 在 CV 里的应用--De(tection) TR(ansformer) 是 transformer 在 Detection 里的成功应用。但是它的收敛很慢（需要训很多 epoch）而且有限的feature spatial resolution。
2. 为何成功，标志/准是什么？比 DETR 的效果要好，尤其在小的物体上，而且epoch可以减小10倍。
3. 在前人基础上的关键创新是什么？ Attention 只考虑在给定的 reference 周围的一个采样结合上做
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？