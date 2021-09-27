1. 在解决的是什么问题？想在 CV 领域里复制 transformer 的优势。
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？发现来了之前 transformer 在 CV 里不好使的原因：没有使用大数据集。只用了 transformer，没有用 CNN 等，而且 transformer 的结构和 NLP 的几乎相同，没有特殊之处
4. 关键结果有哪些？在下游任务上表现良好
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？

## TODO
1. 4实验里，看看三类任务上的效果
2. 看看 EQ 1-4 各自啥意思？

## 简介
在图片上直接使用了标准的 transformer。为了这样做，把图片分成了 patches，然后通过 linear embeddings 来把这些 patch 作为 Transformer 的输入。这样目的是让 transformer 迁移过来后，改动尽量小。图片分割之后的 patches，被当作跟 NLP 里 tokens(词组) 同样的用法。使用有监督的方式来在分类任务上进行训练

Transformer 缺乏一些归纳性的偏见(inductive biases)，跟CNN相比，比如：转换不变性(translation equivariance and locality)，所以如果训练时的数据量不够大，就很难很好地泛化

但是挡在大数据集(14M-300M图片）上训练后，就胜出了。

## 实现

参考的 timm 中 vit 实现：

```
self.cls_token = nn.Parameter(torch.zeors(1, 1, embed_dim)) # 它是一个学来的参数
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+self.num_tokens, embed_dim)) # (1, 196+1, 768), 也是学来的参数，为了让 transformer 对位置有区分。否则输入的 patch 互换，结果是不变的，显然不行

# forward_features
x = self.patch_embed(x)
cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
x = torch.cat((cls_token, x), dim=1)
x = x + self.pos_embed # 为啥是加呢？而不是乘积啥的？
x = self.pos_drop(x) # 这里及以上都是对图片做处理

x = self.blocks(x)
```

## 问题
1. 题目中的 transformers for image recognition at scale. 这个 at scale 代表什么？是说问题规模，还是说较多数量的任务
2. 有其他效果较好的 CNN+ViT 的论文吗？
3. 自注意力和 transformer 是啥关系？
4. pretrain时，也只是大数据集在分类上训练吗？而 finetune 时有检测、分割等？
5. p1 里提到的：when trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of . 这里 strong regularization 是什么意思？
6. Fig1 里，1，2 等格子隔壁的空白格子代表什么？
7. MLP 和 卷积是啥关系？

