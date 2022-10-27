### 主逻辑在 mesh_profiling.py 里的 `estimate_hlo_module_cost` 里
看到这里了：https://github.com/alpa-projects/alpa/blob/main/alpa/pipeline_parallel/stage_profiling.py#L416
## Profile 一台主机
结果如下：

```
        # Cost dictionary for communication primitives.
        # Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
        # The elements in the list is sorted according to the size (ascending).
        self.all_gather_cost_dict = defaultdict(list)
        self.all_reduce_cost_dict = defaultdict(list)
        self.all_to_all_cost_dict = defaultdict(list)
        self.reduce_scatter_cost_dict = defaultdict(list)
        self.available_memory_per_device = None

        # Cost dictionary for computation primitives.
        # Reuse the same data structure.
        # Dict[Tuple(None, dtype)] -> List[Tuple(flop_count, time)]
        self.dot_cost_dict = defaultdict(list)
        self.conv_cost_dict = []

        # Cost dictionary for specific operators
        # Dict[op_info] -> double
        self.op_cost_dict = []
```
计算就在单卡上跑，那通信是主机内和主机间都包括了？而且会跟参与的规模有关呢? 对，所以是一个各种条件都尝试一遍的profile，比如下面通信时的规模，是可以递增到要求的intra、inter  node 最值：
```
        all_specs = enumerate_all_collective_spec(num_hosts,
                                                  num_devices_per_host,
                                                  max_comm_size_intra_node,
                                                  max_comm_size_inter_node)
```
最后存放的也是以规模为key的带宽信息


## xla 上的使用举例
可以参考： `_compile_profiling_executable_while_loop`，里面就是用 `jax._src.lib`里的 `xla_bridge` 和 `xla_client` 构建想要运行的网络、编译


