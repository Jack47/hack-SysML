tensor model parallel + pipeline model parallel 的例子在：
*\_distributed\_with\_mp.sh 里

pipeline model parallel 参数：
pipeline-model-parallel-size 2

split 949,50,1 // 数据划分为三个：training/validation/test 集合
global-batch-size 16
micro batch size 4


tensor model parallel 参数：


## TODO
1. 看如何用几行代码实现的 tensor 级别并行，而且是在 g 和 f 那里要all-reduce 一下
