## sampling
### 1. greedy search
temperature: 控制采样时的随机性程度，比如 0 就是追求确定性，采用 greedy search（每次都选择概率最高的单词）。贪心策略的好处是猜的很快，但有时句子会变得不自然，因为总选择可能性最大的，可能会错过更好、更有趣的选择。

### 2. use_beam_search(束搜索): 
是否用 beam search, 比如 beam_width=3，那么会选择三个单词中概率最高的那个。它不只考虑一个最可能的词，而是保留多个可能性，然后再缩小选择范围。比如我想吃，后面可能是香蕉、苹果、巧克力都不错。

### 3. 概率空间采样(Sampling from a probability distribution)
就像是投骰子决定下一个词。每个词都有可能性，但不一定选最可能的词，而是根据概率随机选择。

top_p 代表 top_p 的概率，比如 0.95，那么就是随机从概率大于等于 0.95 的单词。

top_k

遇到的问题：上述两个概率之一都没有设置，导致采样出低概率的 token。因为是在整个词表空间里参与采样。