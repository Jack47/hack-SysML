## Memory Management

PyTorch 使用 memory allocator 来加速显存分配。能够无须设备同步就可以快速分配显存。torch 会在 reserved 的部分里，分配出来一部分。所以通过 nvidia-smi 看到的总显存占用，分为几部分：

total(nvidia-smi) = torch allocator reserved + torch framework consumed(cuda ctx, cuda rng state, cudnn ctx, cufft plans, other libraries used)

torch allocator reserved = torch allocator allocated + torch allocator cached # empty_cache 会释放这些没被用到的显存，方便其他程序使用这段显存

torch.cuda.* 里包含的，只是 torch allocator 里的内容。而非 pytorch 分配的部分，是没有包含的。一般而言，可能这部分有 200MB~2GB

而 memory\_summary 或者 memory\_states 里展示的，也是类似上述的信息：


```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |  146836 KB |    4261 MB |     847 GB |     847 GB |
|       from large pool |  107268 KB |    4228 MB |     843 GB |     843 GB |
|       from small pool |   39568 KB |      48 MB |       4 GB |       4 GB |
|---------------------------------------------------------------------------|
| Active memory         |  146836 KB |    4261 MB |     847 GB |     847 GB |
|       from large pool |  107268 KB |    4228 MB |     843 GB |     843 GB |
|       from small pool |   39568 KB |      48 MB |       4 GB |       4 GB |
|---------------------------------------------------------------------------|
| GPU reserved memory   |    5176 MB |    5176 MB |    6340 MB |    1164 MB |
|       from large pool |    5124 MB |    5124 MB |    6274 MB |    1150 MB |
|       from small pool |      52 MB |      52 MB |      66 MB |      14 MB |
|---------------------------------------------------------------------------|
| Non-releasable memory |  619115 KB |     839 MB |  351196 MB |  350592 MB |
|       from large pool |  617723 KB |     833 MB |  345883 MB |  345280 MB |
|       from small pool |    1392 KB |      13 MB |    5313 MB |    5312 MB |
|---------------------------------------------------------------------------|
| Allocations           |     595    |     820    |   71826    |   71231    |
|       from large pool |      31    |     150    |   30693    |   30662    |
|       from small pool |     564    |     675    |   41133    |   40569    |
|---------------------------------------------------------------------------|
| Active allocs         |     595    |     820    |   71826    |   71231    |
|       from large pool |      31    |     150    |   30693    |   30662    |
|       from small pool |     564    |     675    |   41133    |   40569    |
|---------------------------------------------------------------------------|
| GPU reserved segments |     106    |     106    |     125    |      19    |
|       from large pool |      80    |      80    |      92    |      12    |
|       from small pool |      26    |      26    |      33    |       7    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |      73    |     107    |   41956    |   41883    |
|       from large pool |      11    |      74    |   22126    |   22115    |
|       from small pool |      62    |      76    |   19830    |   19768    |
|===========================================================================|
```

caching allocator 会影响内存检查工具如 cuda-memcheck，所以为了使用 cuda-memcheck 来debug 显存错误，一般需要设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 来关闭 caching （此时速度会急剧变慢）

## 可以通过环境变量来控制 caching allocator：
`max_split_size_mb`

`roundup_power2_divisions`

`garbage_collection_threshold`: 能主动帮助回收不再被使用的 GPU 显存，避免开销大的 sync-and-reclaim-all 操作(release\_cached\_blocks)。GC 的作用是优先释放老的、未被使用的。比如设置为80%
