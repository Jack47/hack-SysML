## 问题
1. BN 这类需要全局数据时图的场景下怎么做？all_reduce(sum, mean/variance) ? 这里就会同步阻塞住

