包含三部分：

1. 训练pipeline的自动容错和恢复机制 (Transform Operator and Launcher: TOL)：控制训练任务的生命周期，基于有限状态机来管理训练任务的生命周期。包括启动、热身、执行、验证和恢复。提供了动态和自动化的容错恢复。
2. 训练任务的多维度指标异常检测系统(The training task multi-dimensional metric automatic anomaly detection system: Transom Eagle Eye(TEE): 任务监控和异常汇报，TEE 把检测到的异常汇报给 TOL，自动进入容错策略来消除异常节点，重启训练任务；训练时间减少 28%。
3. 恢复技术：Transom Checkpoint Engine(TCE)：TCE 提供的异步保存和加载函数缩短了容错的开销。提速20倍：核心是替换 pytorch save & load。


多方面挑战：

1. 因为硬件和软件问题而导致训练 LLM 时经常出问题
2. 上述问题发生后，定位 LLMs 的问题很有挑战性：需要个把小时甚至更久才行
3. LLMs 的训练任务包含显著的恢复开销

## IV 系统设计

### TCE

关键特性：

**Ultra-low latency** : 每个 Host 上有一个 server，它负责代理 ckpt load 和 save 请求，先写入到内存，然后再异步做持久化

**优秀的容错机制** : 内存容易丢失，所以需要容错，尤其是某个容器被重新调度，会丢失掉内存 cache。每个 TCE server 会把自己的 ckpt cache 放到相邻的下一个节点做备份。TCE server 会通过从上一个 backup 节点拿取备份数据来恢复

**内存使用的灵活性** : 使用 cache eviction 策略。可以设定最大的内存使用量


```
from transomSnapshot.engine import engine

....

model = engine.load(path) # replace torch.load
engine.save(model, path) # replace torch.save
```

理论性能分析：

写入：P是参数量，那么:

1. 2P 的参数量，总共有 8N/DP ranks(PP*TP) 分片保存他们（DDP 内部，只需要一个人保存，然后启动时广播出去）。
2. 优化器状态：假设是 adamw 优化器，大小是 12P 字节存储。而优化器状态会使用全局的 ZeRO 拆分

那么每个 rank 需要保存的数据是： 2P/(PP*TP) + 12P/8N

假设参数是

## 问题
1. 为什么写入速度这么快？
175B，TP=PP=8, DP=2，此时是分 8*2=16 个人写入，需要 4min 30s，而 TCE 里是 10.6s。因为是写入本机内存(DDR4: 20GBps) vs 写入网络存储(10Gbps)之间的速度差异，而且网络存储是大家共享的，而本机内存是很近的，而且随着节点数量增多，网络存储iops是固定的，随着人多而每个人的写入速度降低
上面的例子里，每个rank需要保存的数据大小： `175*2Bytes/16=21.8G + 175*12Bytes/128=16.4=38.2G`，速度是 4.5分钟，那么140MB/s，即每个 rank 上速度是这样的，而TCE里只需要10s就可以写入到内存里：3.82GB/s
2. 当发现异常的节点后，怎么处理的，是全部重启还是部分重启？
3. TCE 里为什么需要一个 MySQL
