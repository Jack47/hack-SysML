DINOv2 比第一代更准确（性能好），而且无须 fine-tuning，可以适合用来做许多 CV 任务下的 backbone；由于使用了自监督，DINOv2 可以从图片集里学习。可以学习feature，比如深度预估，这个是当前方法没法做到的。比起 text 或者 caption 对这种方式能学到更多。

实验发现在分类，分隔，图片召回上都有很强能力。

## 客服 image-text 预训练里的限制
从 DINO 到 DINOv2 需要克服以下困难：

1. 创建大的 curated 的训练数据集
2. 提高训练和实现的算法
3. 设计一个函数蒸馏的 pipeline
