来自：https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4

符号：

训练 transformer 模型所需的计算是 C，它的参数量是 N，训练所需的 tokens 数量是 D

那么就似地： C ～= 6ND，考虑 checkpoint 技术的情况下，大概会是 8ND

此时训练所需时间 T = C/集群的吞吐。而集群吞吐=GPU卡个数*每个上达到的 TFLOPs。平常我们经常说的训练达到的 TFLOPs 就是每个卡上的计算吞吐。