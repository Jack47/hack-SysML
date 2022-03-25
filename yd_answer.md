# 三个问题的回答

## 1.显存分配的大致阶段

在malloc函数中调用`get_free_block(params)`获取内存池中的最小可用块。  
如果失败（未在内存池中找到合适的块），则调用`alloc_block(params, false/true)`函数分配显存。
```cpp
    //尝试分配新的块
    block_found = alloc_block(params, false)
    //尝试释放已经缓存(cached)的（一个或多个）（oversized）块后分配
    || (release_available_cached_blocks(params) 
    &&alloc_block(params, false))
    //尝试释放全部已经缓存(cached)的（non-split）块后分配
    || (release_cached_blocks() 
    && alloc_block(params, true));
```
在`alloc_block()`函数中，调用  
```
	cudaMallocMaybeCapturing(&ptr, size);
```
进行实际的分配。  
并用
```
	p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
```
初始化对显存管理的块。

## 2.memory fragmentation 产生的原因是什么？
(1) 在malloc函数中，(用1.中方法) 得到可以分配的块后，块可能会比需要分配的空间(round_size)大，则需要对块进行切割。
```
    if (should_split(block, size)) {
      remaining = block;

      block = new Block(device, stream, size, &pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      bool inserted = pool.blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
 ```
此时，剩余的块(remaining)的size减小。对于越小的块，可能越难以被分配，此时会留存在pool中但是难以被后续alloc，造成内存碎片。  
(2)在free中，会对不使用内存进行释放，`	free_block(block);`

```
    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
      	net_change_inactive_split_blocks -= 1;
      	net_change_inactive_split_size -= subsumed_size;
      }
    }
```
但如果前后的邻居内存块并没有被释放，会导致内存块中的空间无法被合并(merge)后再分配，造成内存碎片。

(3)在代码逻辑中，为了减少内存碎片，小请求（<1MB）会分配2MB的块。在1MB到10MB的请求会分配20MB的块（如果未在池中找到合适的块）。
```
static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {//<1MB
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {//<10MB  
      return kLargeBuffer;
    } else {//>10MB，round
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }
```
超大（oversized，>max_split_size，大于200MB？）的块不会被分割，除非请求就是超大的(>max_split_size)且在分割后剩余的块小于20MB(kLargeBuffer)。
```
bool get_free_block(AllocParams& p)
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
```

## 3.如何知道当前要释放的这个 block，它在显存地址空间里，前后的两个邻居块的地址？
对于`struct Block`有成员
```
	Block* prev; // prev block if split from a larger allocation
	Block* next; // next block if split from a larger allocation
```
指向前后邻居块。  
成员
```
	void* ptr; // memory address
```
则指向每个块实际被分配的memory地址。

## 4. 可以试着解释一下max_split_size_mb这个参数的作用嘛？
```
void parseArgs() 
//在参数解析函数中
for (auto option : options)
//对某个option进行解析
if (kv[0].compare("max_split_size_mb") == 0)
//如果成立，说明option解析出来的是对max_split_size的描述，而且是以MB为单位
size_t val2 = stoi(kv[1]);
//val2就是解析出max_split_size_mb的实际值
TORCH_CHECK(val2 > kLargeBuffer / (1024 * 1024)，...）
//需要保证max_split_size>kLargeBuffer，***/ (1024 * 1024)的原因是kLargeBuffer以byte为单位存储
val2 = std::max(val2, kLargeBuffer / (1024 * 1024));
//如果val2 <= kLargeBuffer / (1024 * 1024)，则将val2赋值为kLargeBuffer / (1024 * 1024)
val2 = std::min(val2, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
//并且m_max_split_size还有std::numeric_limits<size_t>::max()的限制（size_t类型的最大值）
m_max_split_size = val2 * 1024 * 1024;
//对处理完上述限制的val2，赋值给m_max_split_size
```
