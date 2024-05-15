

参考资料：
1. The FLOPs Calculus of Language Model Training https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
介绍了 throughput * training-time-per-iter = 6N*D . N: Model Size, D 是 Tokens 数量。其中 FWD 是2N 的计算量，而 BWD 是 4N 计算量
2. https://stanford-cs324.github.io/winter2022/assets/pdfs/Scaling%20laws%20pdf.pdf
