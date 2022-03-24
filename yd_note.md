# 笔记
对[pytorch显存分配策略](https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp)代码的笔记和一些注释  
/*显存分配是用流组织的，流中的块被释放后可以在同一个流中重新分配，但是不可以在不同流中使用。  
**找可用的最小块，并分割。**  
如果没有，则用cudaMalloc  
如果失败，则释放已经缓存的块并尝试重新分配  
如果失败，则释放全部缓存的没有被分割的块并尝试重新分配

**大（>1MB）和小的分配被存储在不同的内存池中。**  
小的请求会被打包进2MB的缓存中  
大的请求则会使用最小可用块或让cudaMalloc分配新的块

**为了减少内存碎片，在1-10MB之间的请求，如果没有可用块，则会分配并分割一个20MB的块**

**为了进一步减少碎片，>=200MB的块不会被分割。这些巨大的缓存块将会满足差20MB以内的请求。**

如果内存段用于多个流，则需要同步。

recordStream()帮助插入同步。*/  
//////////////////////////////////////////////  
```
struct Stat
{
	current;//当前
	peak;//顶峰
	allocated;//分配
	freed;//释放
}
```

`StatTypes=array<bool>`



//更新stat（增加或减少amount）  
`void update_stat(Stat& stat, int64_t amount)`

//重置（累计的）stat  
`void reset_accumulated_stat(Stat& stat) `  
//reset allocated;  
//reset freed;

//重置（顶峰）  
`void reset_peak_stat(Stat& stat)`  
//reset peak;  

//对stat_types调用f  
`void for_each_selected_stat_type(const StatTypes& stat_types, Func f)`

//对stat_array中的内容更新  
`void update_stat_array(StatArray& stat_array, int64_t amount, const StatTypes& stat_types) `


**//block内存块池**
```
struct BlockPool {
  BlockPool(
      Comparison comparator,
      bool small,
      PrivatePool* private_pool = nullptr)
      : blocks(comparator), is_small(small), owner_PrivatePool(private_pool) {}
  std::set<Block*, Comparison> blocks;//set记录block
  const bool is_small;//是大还是小池
  PrivatePool* owner_PrivatePool;//私人池？
};
```

**//内存块**
```
struct Block {
  int device; // gpu
  cudaStream_t stream; // allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  BlockPool* pool; // owning memory pool
  void* ptr; // memory address
  bool allocated; // in-use flag
  Block* prev; // prev block if split from a larger allocation
  Block* next; // next block if split from a larger allocation
  int event_count; // number of outstanding CUDA events
  int gc_count; // counter for prioritizing older / less useful blocks for
                // garbage collection //计数器用于在垃圾回收时排序
bool is_split() const {
    	return (prev != nullptr) || (next != nullptr);
  	}//判断块是否被分割
};
```


//块比较，先比较是否在同一个流中，如果在，比较块的大小(<)  
`static bool BlockComparator(const Block* a, const Block* b)`

//对size 调整格式，bytes调整为bytes/KB/MB/GB  
`static std::string format_size(uint64_t size)`

**//分配器参数**
```
struct AllocParams {
    int device() const {
    return search_key.device;
  }
  cudaStream_t stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;//key用于查询
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {false};
  cudaError_t err;
};
```


//CUDA graphs helper  ?
```
struct PrivatePool
{
// Number of live graphs using this pool
  int use_count;
// Number of unfreed cudaMallocs made for this pool. When use_count and
  // cudaMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int cudaMalloc_count;
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance- critial though,
  // I'd rather not add more logic to it.
  BlockPool large_blocks;
  BlockPool small_blocks;
};
```

//hash id
```
struct MempoolIdHash {};
```

//capturing?
```
cudaError_t cudaMallocMaybeCapturing(void** p, size_t size)
```

**//缓存分配参数**
```
class CachingAllocatorConfig {
 public:
  static size_t max_split_size() ;

  static double garbage_collection_threshold() ;

  // This is used to round-up allocation size to nearest power of 2 divisions.
  static size_t roundup_power2_divisions();

 private:
  static CachingAllocatorConfig& instance() {
    return *s_instance; //new instance and parseArgs
  }
```

**//config**
```
  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_roundup_power2_divisions(0),
        m_garbage_collection_threshold(0) {}
  size_t m_max_split_size;
  size_t m_roundup_power2_divisions;
  double m_garbage_collection_threshold;
  ```
  
  
**//解析PYTORCH_CUDA_ALLOC_CONF的/参数？**  
//包含m_max_split_size m_roundup_power2_divisions m_garbage_collection_threshold等
```
  void parseArgs() 
  ```
 **//device alloctor**  
 ```
class DeviceCachingAllocator
{
  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB,大池
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller，小池
  BlockPool small_blocks;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<Block*> active_blocks;
  
    // outstanding cuda events
  ska::flat_hash_map<
      cuda::CUDAStream,
      std::deque<std::pair<cudaEvent_t, Block*>>>
      cuda_events;
      
    // record used memory.
  size_t total_allocated_memory = 0;

  size_t allowed_memory_maximum = 0;

  bool set_fraction = false;

// Private pools for CUDA graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;

  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  // Maps a capturing stream to its assigned private pool,
  // in case we want multiple captures to share the same pool
  ska::flat_hash_map<CaptureId_t, MempoolId_t> capture_to_pool_map;
}

  DeviceCachingAllocator()
      : large_blocks(BlockComparator, /*is_small=*/false),
        small_blocks(BlockComparator, /*is_small=*/true) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
  }


//device allocator.malloc
  Block* malloc(int device, size_t size, cudaStream_t stream) {

    size = round_size(size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);

    // First, try to get a block from the existing pool.
    bool block_found =
        // Search pool
        get_free_block(params)//直接在池中获取
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && get_free_block(params));

    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {//未找到，
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(//尝试进行垃圾回收
              CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_blocks();
      }
      // Attempt allocate
      block_found = alloc_block(params, false)//直接分配
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          || (release_available_cached_blocks(params) &&
              alloc_block(params, false))//尝试释放部分超大块缓存后分配
          // Free all non-split cached blocks and retry alloc.
          || (release_cached_blocks() && alloc_block(params, true));
//释放全部未分割缓存块后分配
    }
    
    	// "total capacity": total global memory on GPU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CUDA API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      
      
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (should_split(block, size)) {//对找到的块进行切分}
    else if(already_split) {//not should_split(block, size)}
    
    block->allocated = true;
    }

//device allocator free
  void free(Block* block) {
  block->allocated = false;
  update_stat();
  if (!block->stream_uses.empty())
  	needs_events_deferred_until_no_capture.push_back(block);
    //or
   insert_events(block);
  else
  	free_block(block);
 }
 
 
 
//找到block所在的完整allocation，以及分配的内存大小
  void* getBaseAllocation(Block* block, size_t* outSize) {
    
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;

	 size += block->size;
        
 
    *outSize = size;
    }
    return basePtr;
  }
  
  //记录block被使用的流
  void recordStream(Block* block, cuda::CUDAStream stream) {
    size_t device_free;
    size_t device_total;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }
  
  
/** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
  }

  /** release cached blocks to the system allocator **/
  void emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks();
  }

  /** Retrieves info (total size + largest block) of the memory cache **/
  void cacheInfo(size_t* total, size_t* largest) {
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
      }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
  }



/** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() const {
  std::vector<SegmentInfo> result;
  const auto all_blocks = get_all_blocks();
  ...
  std::sort(result);

  return result;
  }
  
  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
    static size_t roundup_power2_next_division(size_t size, size_t divisions) { }
    
    
  //由size返回round_size
  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      auto divisions = CachingAllocatorConfig::roundup_power2_divisions();
      if (divisions > 0 && size > (kMinBlockSize * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
      }
    }
  }
  
    // See Note [Interaction with CUDA graph capture]？

  // Called by CUDAGraph::capture_begin
  void notifyCaptureBegin(CaptureId_t graph_id, MempoolId_t mempool_id) {}
  
  // Called by CUDAGraph::capture_end
  void notifyCaptureEnd(CaptureId_t graph_id) {}
  
    // Called by CUDAGraph::reset
  void notifyCaptureDestroy(MempoolId_t mempool_id) {}
  
  
  private:
  // All private methods do not acquire the allocator mutex.

std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(
        blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(
        blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    for (const auto& gp : graph_pools) {
      blocks.insert(
          blocks.end(),
          gp.second->small_blocks.blocks.begin(),
          gp.second->small_blocks.blocks.end());
      blocks.insert(
          blocks.end(),
          gp.second->large_blocks.blocks.begin(),
          gp.second->large_blocks.blocks.end());
    }
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }


/** moves a block into a pool of cached free blocks */
	//将一个被free的块加入池中
  void free_block(Block* block) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());


        try_merge_blocks(block, merge_candidate, pool);


    active_blocks.erase(block);
    
    update_stat()
    });
  }
  
  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {}
  
  BlockPool& get_pool(size_t size, cudaStream_t stream) 
  {
      if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }

  }


StatType get_stat_type_for_pool(const BlockPool& pool) {
    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
  }


  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }


//根据size确定分配的内存块大小
  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }
  
  
  //在内存池中获得合适的内存块
  bool get_free_block(AllocParams& p)
  {
  auto it = pool.blocks.lower_bound(&p.search_key);//二分查找找到符合的block
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
  }
  
  
  //直接get_free_block(params)失败，则需要尝试callback后再get
  bool trigger_free_memory_callbacks(AllocParams& p){}

  
  //垃圾回收，free unused cached blocks，回收未被使用（分割）的缓存的块
  void garbage_collect_cached_blocks() {
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

      // No need to trigger GC yet，已经分配内存小于阈值，不回收
    if (total_allocated_memory <= gc_threshold) {
      return;
}

    while (gc_reclaimed < target_size && block_freed == true &&
           freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;
      ...
      release_block(block);
      }
  }


//尝试在池中获取失败，进行真实的内存分配
  bool alloc_block(AllocParams& p, bool isRetry) {
  	p.err =cudaMallocMaybeCapturing(&ptr, size);
	//成功分配
    if (p.pool->owner_PrivatePool) {
      // The block is for a CUDA graph's PrivatePool.
      p.pool->owner_PrivatePool->cudaMalloc_count++;
    }
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
        // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    return;
    
  }

/** Free one or more oversize blocks to the system allocator.  But only enough
   * **/
  /** to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p) {
  
      key.size = (key.size < CachingAllocatorConfig::max_split_size())
        ? CachingAllocatorConfig::max_split_size()
        : key.size;//设置二分查找的key大小
    auto it = pool.blocks.lower_bound(&key);

	if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest

	while ((totalReleased < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
             ...
             release_block(*cur);
             }
    return;
    }


bool release_cached_blocks() {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events();

    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks);
    release_blocks(small_blocks);
}

//release one block
void release_block(Block* block) {
    C10_CUDA_CHECK(cudaFree((void*)block->ptr));

	for_each_selected_stat_type(update_stat());
    
    pool->blocks.erase(block);
    delete block;
  }

void release_blocks(BlockPool& pool) {
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

cudaEvent_t create_event_internal() {}

void free_event_internal(cudaEvent_t event) {
    C10_CUDA_CHECK(cudaEventDestroy(event));
  }


void synchronize_and_free_events() {
    // Synchronize on outstanding events and then free associated blocks.
insert_events_deferred_until_no_capture();
free_event_internal(event);

block->event_count--;
if (block->event_count == 0) {
free_block(block);
        
   }
   cuda_events.clear();
}

void insert_events(Block* block) {
	for (auto& stream : streams) {
      C10_CUDA_CHECK(cudaSetDevice(stream.device_index()));

	cudaEvent_t event = create_event_internal();
    block->event_count++;
    cuda_events[stream].emplace_back(event, block);
}

void insert_events_deferred_until_no_capture() {
    if (C10_UNLIKELY(needs_events_deferred_until_no_capture.size() > 0)) {
      for (auto* block : needs_events_deferred_until_no_capture) {
        TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
        insert_events(block);
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }
  
  void process_events() {
    insert_events_deferred_until_no_capture();
	// Process outstanding cudaEvents.
}


  // Accumulates sizes of all memory blocks for given device in given pool
  void cache_info_aux(const BlockPool& pool, size_t* total, size_t* largest) {}
  
};//end of DeviceCachingAllocator
```
**//THC集装箱？**
```
class THCCachingAllocator {

  // allocated blocks by device pointer
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

  public:
//THC中包含DeviceCachingAlloctor的vector
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

Block* get_allocated_block(void* ptr, bool remove = false) {}

void init(int device_count) {//按照设备数量初始化多个device分配器
    const auto size = static_cast<int64_t>(device_allocator.size());
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
      }
    }
  }
  
    /** allocates a block which is safe to use from the provided stream */
    
  //malloc of THC
void malloc(void** devPtr, int device, size_t size, cudaStream_t stream) {    
	Block* block = device_allocator[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;//返回参数，malloc分配的内存
  }

//free(ptr) of THC
  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    device_allocator[block->device]->free(block);
  }


//对某个device设置内存限制（Fraction）
  void setMemoryFraction(double fraction, int device) {
	device_allocator[device]->setMemoryFraction(fraction);
}



void* getBaseAllocation(void* ptr, size_t* outSize) {
	Block* block = get_allocated_block(ptr);
	//返回被（base）分配块的内存首指针（basePtr）
	return device_allocator[block->device]->getBaseAllocation(block, outSize);
}

void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) {
	Block* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
    device_allocator[block->device]->recordStream(block, stream);
  }
  
  
  std::vector<SegmentInfo> snapshot() {
    std::vector<SegmentInfo> result;
    for (auto& da : device_allocator) {
      auto snap = da->snapshot();
      result.insert(result.end(), snap.begin(), snap.end());
    }

    return result;
  }
};//end of THC
```
**//定义一个THC**
```
THCCachingAllocator caching_allocator;
```

// Returns whether to force all allocations to bypass the caching allocator and go straight to cudaMalloc.  This setting is useful when debugging GPU memory errors, since the caching allocator foils cuda-memcheck.
```
bool forceUncachedAllocator() {
  static bool force_uncached =
      getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  return force_uncached;
}
```
```
static void uncached_delete(void* ptr) {
  C10_CUDA_CHECK(cudaFree(ptr));
}
```

// NB: I decided not to fold this into THCCachingAllocator, because the latter has a lot more methods and it wasn't altogether clear that they should actually be publicly exposed  
**//CUDA allocator，是对THC的调用**
```
struct CudaCachingAllocator : public Allocator {??
DataPtr allocate(size_t size) const override {
    if (forceUncachedAllocator()) {
        cudaMalloc(&r, size)
    }
    if (size != 0) {
      caching_allocator.malloc(
          &r, device, size, cuda::getCurrentCUDAStream(device));
    }
    return {r, r, &raw_delete, Device(DeviceType::CUDA, device)};
}

DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &raw_delete;
    }
	}
};//end of cuda allocator
```

**//定义一个CudaAlloctor**
```
CudaCachingAllocator device_allocator;
```
**//caching_allocator的部分函数**
```
Allocator* get(void) {
  return &device_allocator;
}

void init(int device_count) {
  caching_allocator.init(device_count);
}

void setMemoryFraction(double fraction, int device) {
  caching_allocator.setMemoryFraction(fraction, device);
}

void emptyCache(void) {
  caching_allocator.emptyCache();
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
  caching_allocator.device_allocator[dev_id]->cacheInfo(
      cachedAndFree, largestBlock);
}

void* getBaseAllocation(void* ptr, size_t* size) {
  return caching_allocator.getBaseAllocation(ptr, size);
}

void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) {
  caching_allocator.recordStream(ptr, stream);
}

std::mutex* getFreeMutex() {
  return caching_allocator.getCudaFreeMutex();
}


//assert
static inline void assertValidDevice(int device){}

//stats
DeviceStats getDeviceStats(int device) {
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->getStats();
}

void resetAccumulatedStats(int device) {
  assertValidDevice(device);
  caching_allocator.device_allocator[device]->resetAccumulatedStats();
}

void resetPeakStats(int device) {
  assertValidDevice(device);
  caching_allocator.device_allocator[device]->resetPeakStats();
}

std::vector<SegmentInfo> snapshot() {
  return caching_allocator.snapshot();
}
```

**// CUDAGraph interactions**
```
void notifyCaptureBegin(
    int device,
    CaptureId_t graph_id,
    MempoolId_t mempool_id)

void notifyCaptureEnd(int device, CaptureId_t graph_id) 

void notifyCaptureEnd(int device, CaptureId_t graph_id)

void notifyCaptureDestroy(int device, MempoolId_t mempool_id)

```

**// CUDA IPC（interprocess communication）**
```
namespace {
std::mutex IpcMutex;
ska::flat_hash_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
} // namespace

std::shared_ptr<void> getIpcDevPtr(std::string handle) {}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(
      &r, device, nbytes, cuda::getCurrentCUDAStream(device));
  return r;
}

void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, device, nbytes, stream);
  return r;
}

void raw_delete(void* ptr) {
  caching_allocator.free(ptr);
}

```
//////////////////////////end/////////////////////////////








