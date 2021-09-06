1. 在解决的是什么问题？进一步提高进度 
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？使用了 Region Proposal Network，它是一个 CNN，而且与下游的检测网络共享卷积 feature，所以训练和推理的代价很低。提出了统一的，运行速度接近实时的算法。RPN 提高了 Region proposal的质量
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？


## 疑问
1. 4.1 里的 One-Stage vs Two-Stage Proposal + Detection 有何区别？
2. IoU 相关

## 启发
1. Fast RCNN 和 Faster-RCNN 联合训练：让 RPN 和 Fast R-CNN 一起共享 feature
2. Multiple scale anchors 是让 Fast R-CNN 和 Faster 共享feature 的关键，无需额外开销
3. 为了让 RPNs 和 Fast R-CNN 里的统一起来，使用训练方法来可以fine-tune RPN 任务和目标检测任务
