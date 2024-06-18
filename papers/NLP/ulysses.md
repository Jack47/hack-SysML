DeepSpeed Ulysses ，自己论文里又叫 DeepSpeed sequence parallelism
对输入的 sequence 在 length 维度上进行切分。然后进行每个 head 上的 attention 计算。之后，通过 all2all 把输出转为给后续 op(MLP,LN 等） 所需的 sequence(N/P) 维度并行。

## 3

### 3.2 通信分析
all2all 上 M 大小的消息在 P GPU 上传递，此时每个 link 上通信量为 M/P。

每个 transformer layer 上：QKV projection 后(embedding) 的聚合通信大小为 3Nh（attention 计算前），output context projection 之后的 all2all 通信量是 Nh
因此每个 link 上的 DS sp 涉及的聚合通信量为：4Nh/P （即复杂度是 O(N/P)）。

对比而言，已有的比如 MLM 里的通信量随着 N 的增大会线性增加，而与 P 无关(因为就没在 seq 维度上切分，除了 LN 和 Dropout）。 MLM 里每个 transformer layer 上执行两次 all-gather，通信量是 Nh，两次 reduce-scatter，通信量是 Nh。即使 P 很大的情况下，size M 的 ag 和 rs 的通信量依然是 M，而非 M/P。
### 3.3 显存效率
ZeRO-3 下，Optimizer states 和 gradients 可以切分到 DP和 SP ranks 上。

## 4 评估
