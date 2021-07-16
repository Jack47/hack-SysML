tensor model parallel + pipeline model parallel 的例子在：
*_distributed_with_mp.sh 里

pipeline model parallel 参数：
pipeline-model-parallel-size 2

split 949,50,1 // 数据划分为三个：training/validation/test 集合
global-batch-size 16
micro batch size 4


tensor model parallel 参数：
