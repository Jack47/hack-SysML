
[stochastic rounding](https://medium.com/@minghz42/what-is-stochastic-rounding-b78670d0c4a)

## [Embedding](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture): 
An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. It caputres some of the semantics of the input by placing **semantically similar inputs close together** in the embedding space. An embedding can be learned an reused across models. 从一个空间映射到一个向量空间里，而且具备相似的输入距离更近的特点。而且它可以做可视化，有降维的工具和技术，比如 t-Distributed Stochastic Neighbor Embedding (TSNE)

参考资料：
[zhihu Embedding explained](https://zhuanlan.zhihu.com/p/46016518)

Embedding 层可能占整个网络很大(参数量)的比重，所以也可以单独做预训练。它的训练频率可以设定一周或一天等。由于 graph embedding等技术的发展，embedding自身的表达能力也逐步增强，可以直接利用它的相似性，做推荐系统里的召回层。

Embedding 有 User Embedding, Video Embedding 等

Float to
