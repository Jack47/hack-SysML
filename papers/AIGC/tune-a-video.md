1. 在解决的是什么问题？
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## 摘要
有两个发现：
1. T2I 模型可以产生代表动词的静态图片
2. 扩展 T2I 模型

## 介绍
作为 T2I 的见过大量图片-文本数据的模型，已经捕获了很多开放领域概念。把空间上的self attention从一张图片扩展到多张图片，就可以产生跨帧的连续内容。

如果使用大量视频训练，训练开销会很大，我们希望通过在已有的 T2I 模型上通过一个 text-video 对来做预训练。

我们的方法是把 2D 的模型扩展到 latentspace 下的 spatio-temporal(空间-时间）域

看效果能达到主题是一致的，但是整体肉眼效果不好，中间有缺的帧，导致不连续

