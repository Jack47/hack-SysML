包含两种编码格式：E4M3 或者 E5M2。在图片和自研语言任务上都能运行，而且质量和 16bit 混合精度训练质量相当。在 CNNs、RNNs 和 Transformer 模型上都可以。

### 4.1 Training
目前只是把算术密集的算子 - conv 和 matrix multiples，这种涉及到 dot-product 计算的算子替换为 fp8：激活值、权重和激活值的梯度 tensor 这些 GEMM 的输入都当做 fp8.

## 5 总结
使用 FP8 不仅加速和减少训练时候的显存占用，而且简化了 8-bit 推理部署，因为训练和推理用的相同的精度
