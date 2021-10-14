## APIs
### BlockLoad/Store
Block级别，做 thread 里 block I/O 操作的

```
#include <cub/cub.cuh>

  typedef cub::BlockLoad<T, block_dim, ele_per_thread,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;
  
  
  for ()
    BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,
                              REDUCE_FLOAT_INF_NEG);
```

需要关注：

1. block_dim * ele\_per\_thread = 每个 `block` 里要处理的元素数量(threads\-per\-block * ele\_per\_thread)
2. 每个 block 里做的事情都类似，而每个 thread 里处理 ele\_per\_thread 的数据
3. cub's 的 Block 操作通常需要所有线程都执行 collective
