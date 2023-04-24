## Debugging torch.distributed applications
V1.10 之后，提供了 [`torch.distributed.monitored_barrier()`](https://pytorch.org/docs/stable/distributed.html#monitored-barrier) 来代替 .barrier()，目的是rank 0 能知道哪个 rank 没有在规定的 barrier 时间里返回。

```
RuntimeError: Rank 1 failed to pass monitoredBarrier in 2000 ms
 Original exception:
[gloo/transport/tcp/pair.cc:598] Connection closed by peer [2401:db00:eef0:1100:3560:0:1c05:25d]:8594
```

### TORCH_DISTRIBUTED_DEBUG
`TORCH_DISTRIBUTED_DEBUG` 支持 OFF(默认）、INFO、或者 DETAIL。

INFO：

DETAIL：额外记录运行时的统计数据

```
I0607 16:18:58.085681 544067 logger.cpp:344] [Rank 1 / 2] Training TwoLinLayerNet unused_parameter_size=0
 Avg forward compute time: 40838608
 Avg backward compute time: 5983335
Avg backward comm. time: 4326421
 Avg backward comm/comp overlap time: 4207652
I0607 16:18:58.085693 544066 logger.cpp:344] [Rank 0 / 2] Training TwoLinLayerNet unused_parameter_size=0
 Avg forward compute time: 42850427
 Avg backward compute time: 3885553
Avg backward comm. time: 2357981
 Avg backward comm/comp overlap time: 2234674
 ```
