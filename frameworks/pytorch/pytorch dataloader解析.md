## intro

pytorch数据读取，有两个部分，一是dataset本身，二是dataset的加载器，包括三个类：

- DataSet，**提供DataLoder使用的接口**。可以实现序列协议，即__getitem__方法以及__len__方法；或者迭代器协议，即实现__iter__ 方法返回一个迭代器，该迭代器实现__next__ 方法；
- DataLoder以及DataLoaderIter，实现数据的加载。**Combines a dataset and a sampler, and provides an iterable over the given dataset**。其实现__iter__，返回一个_BaseDataLoaderIter类型的对象，可以通过继承_BaseDataLoaderIter并实现__next__方法，定制单进程或多进程加载数据的行为；



dataloader支持以下功能：

- 两种类型的dataset。dataset可分为两类：map-style dataset即继承抽象基类Dataset实现了序列协议的类型，支持**key-value访问**；iterable-style dataset即继承IterableDataset并实现迭代器协议的类型，只支持**顺序访问**；
- 定制数据加载的顺序。对于iterable-style dataset数据加载顺序由迭代器自身实现；对于map-style dataset，torch.utils.data.Sampler 类用于**指定数据加载中使用的 indices的序列**，其是所有定制的sampler的基类，继承它应实现__iter__ 方法以及__len__方法，对sampler进行遍历产生的是 **indices**。常见的sampler有SequentialSampler、RandomSampler和BatchSampler等；
- 加载batch数据。通过参数batch_size, drop_last, and batch_sampler来定制batch大小，是否忽略最后一个batch（无法整除）以及定制batch_sampler的行为。在加载数据时，需要调用**collate_fn****函数来将batch个数据在batch维度进行聚合，若是numpy对象则转换为tensor**（Puts each data field into a tensor with outer dimension batch size）。
- 多进程数据加载数据。将num_workers设为一个正整数时即开启多进程加载数据。对于map-style dataset，**主线程使用sampler生成 indices，接着发送给工作进程进行IO操作**；对于iterable-style dataset每个工作进程都会拿到一个迭代器的复制，这样会导致加载到重复的数据，需要通过 [torch.utils.data.get_worker_info()](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info) 以及 worker_init_fn来进行解决。
- memory pinning。所谓pinned memory即进程的虚拟内存的某个page对应的page frame常驻内存，不会被swap出去，使用pin_memory=True可以将数据加载在pinned memory上。



#  dataloader源码解析

在dataloader.py中关键的四个类：

- DataLoader
- _BaseDataLoaderIter以及其子类 _SingleProcessDataLoaderIter（单线程）和_MultiProcessingDataLoaderIter（多线程）；



### 1. DataLoader

在DataLoader类中，__init__对参数进行合法性检查，根据参数进行一些设置。该类的关键在__iter__方法，在单进程中每次调用__iter__  都返回一个新的迭代器；而在多进程中，若已经创建过迭代器，则reset它再返回，这样可以重复使用worker进程：

```python
def __iter__(self) -> '_BaseDataLoaderIter':
    # When using a single worker the returned iterator should be
    # created everytime to avoid reseting its state
    # However, in the case of a multiple workers iterator
    # the iterator is only created once in the lifetime of the
    # DataLoader object so that workers can be reused
    if self.persistent_workers and self.num_workers > 0:
        if self._iterator is None:
            self._iterator = self._get_iterator()
        else:
            self._iterator._reset(self)
        return self._iterator
    else:
        return self._get_iterator()
```



_get_iterator()根据单进程还是多进程加载返回一个对应类型的新的迭代器：

```python
def _get_iterator(self) -> '_BaseDataLoaderIter':
    if self.num_workers == 0:
        return _SingleProcessDataLoaderIter(self)
    else:
        self.check_worker_number_rationality()
        return _MultiProcessingDataLoaderIter(self)

```



### 2. BaseDataLoaderIter

接着是_BaseDataLoaderIter，主要在其__next__方法，首先是通过_next_data方法获取数据，并返回数据。

而_next_data是子类继承该类来实现的，对于单进程和多进程的数据加载行为不同。

```python
    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                self._reset()
            data = self._next_data()
            self._num_yielded += 1
            if self._dataset_kind == _DatasetKind.Iterable and \
                    self._IterableDataset_len_called is not None and \
                    self._num_yielded > self._IterableDataset_len_called:
                warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                            "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                  self._num_yielded)
                if self._num_workers > 0:
                    warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                 "IterableDataset replica at each worker. Please see "
                                 "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
                warnings.warn(warn_msg)
            return data
```





### 3. SingleProcessDataLoaderIter

_SingleProcessDataLoaderIter 单进程的数据加载。分两步，第一调用sampler的next方法，得到**下一次需要加载的数据的 indices**，第二，调用_dataset_**fetcher.fetch(index)加载数据**。

在_dataset_fetcher.fetch方法中会**collate_fn**函数来将一个batch的数据整合为一个tensor**。**

```python
def _next_data(self):
    index = self._next_index()  # may raise StopIteration
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    if self._pin_memory:
        data = _utils.pin_memory.pin_memory(data)
    return data
```

这样，就实现了一个单进程的for batch in loader的加载数据的逻辑。



### 4. MultiProcessingDataLoaderIter

_MultiProcessingDataLoaderIter多进程的数据加载。利用多进程**对数据进行预取**，主要流程为：

1. 在主进程拿到需要加载的数据的 indices，将其放入某个工作进程的index_queue中；
2. 工作进程从index_queue拿到indices，进行数据加载，将其放入data_queue中；
3. 主进程将一个batch的数据拿到后，返回该数据；

如下图：

```python
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.
```



首先来看该类的__init__ 方法，主要是创建_num_workers个进程以及它们对应的index_queue，以及创建主进程读取数据的data_queue，最后执行reset方法 ：

```python
def __init__(self, loader):
    ......
    
    # 定制worker进程的初始化
    self._worker_init_fn = loader.worker_init_fn
    # 负载均衡用，决定当前indices交给哪个进程加载
    self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
    self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
   ......
    self._index_queues = []
    self._workers = []
    for i in range(self._num_workers):
        # No certainty which module multiprocessing_context is
        index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        # Need to `cancel_join_thread` here!
        # See sections (2) and (3b) above.
        index_queue.cancel_join_thread() # 这是啥？
        # 创建进程并启动 进程运行_utils.worker._worker_loop函数
        w = multiprocessing_context.Process(
            target=_utils.worker._worker_loop,
            args=(self._dataset_kind, self._dataset, index_queue,
                  self._worker_result_queue, self._workers_done_event,
                  self._auto_collation, self._collate_fn, self._drop_last,
                  self._base_seed, self._worker_init_fn, i, self._num_workers,
                  self._persistent_workers))
        w.daemon = True # 默认值为False，如果设为True，代表p为后台运行的守护进程，当p的父进程终止时，p也随之终止，并且设定为True后，p不能创建自己的新进程，必须在p.start()之前设置

        # NB: Process.start() actually take some time as it needs to
        #     start a process and pass the arguments over via a pipe.
        #     Therefore, we only add a worker to self._workers list after
        #     it started, so that we do not call .join() if program dies
        #     before it starts, and __del__ tries to join but will get:
        #     AssertionError: can only join a started process.
        w.start()
        self._index_queues.append(index_queue)
        self._workers.append(w)
    
    if self._pin_memory:
        ......
    else:
        self._data_queue = self._worker_result_queue
    ......
    
    self._reset(loader, first_iter=True)
```



现在来看reset方法，为每个分配给进程的任务分配一个idx（永远自增），通过一个区间表示正在执行的任务[revd_idx, send_idx)以及通过缓冲区task_info，保证读取的batch的有序性：

```python
def _reset(self, loader, first_iter=False):
    super()._reset(loader, first_iter)
    self._send_idx = 0  # idx of the next task to be sent to workers
    self._rcvd_idx = 0  # idx of the next task to be returned in __next__
    # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
    # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
    #                  \ (worker_id, data)   if data is already fetched (out-of-order)
    self._task_info = {} # 也承担着缓冲区的作用 保证取得data的有序性
    # 待完成的任务数量
    self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
    # A list of booleans representing whether each worker still has work to
    # do, i.e., not having exhausted its iterable dataset object. It always
    # contains all `True`s if not using an iterable-style dataset
    # (i.e., if kind != Iterable).
    # Not that this indicates that a worker still has work to do *for this epoch*.
    # It does not mean that a worker is dead. In case of `_persistent_workers`,
    # the worker will be reset to available in the next epoch.
    self._workers_status = [True for i in range(self._num_workers)]
    # We resume the prefetching in case it was enabled
    if not first_iter:
        for idx in range(self._num_workers):
            self._index_queues[idx].put(_utils.worker._ResumeIteration())
        resume_iteration_cnt = self._num_workers
        while resume_iteration_cnt > 0:
            return_idx, return_data = self._get_data()
            if isinstance(return_idx, _utils.worker._ResumeIteration):
                assert return_data is None
                resume_iteration_cnt -= 1
    # prime the prefetch loop 预取_prefetch_factor * self._num_workers个batch
    for _ in range(self._prefetch_factor * self._num_workers):
        self._try_put_index()
        
        
```



对于主进程，它是index_queue的生产者，对应try_put_index方法；它也是data_queue的消费者，对应get_data方法。

try_put_index方法：

1. 首先判断self._tasks_outstanding < self._prefetch_factor * self._num_workers，限制最大的并行任务数；
2. 接着从smpler拿到下一批batch的indices；
3. 然后选取一个可工作的worker进程（这里采用round robin调度）；
4. 将indices发送到其index_queue，_tasks_outstanding待完成的任务数量加一，_send_idx加一，以及保存任务信息；

```python
def _try_put_index(self):
    assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

    try:
        index = self._next_index()
    except StopIteration:
        return
    for _ in range(self._num_workers):  # find the next active worker, if any
        worker_queue_idx = next(self._worker_queue_idx_cycle)
        if self._workers_status[worker_queue_idx]:
            break
    else:
        # not found (i.e., didn't break)
        return

    self._index_queues[worker_queue_idx].put((self._send_idx, index))
    self._task_info[self._send_idx] = (worker_queue_idx,)
    self._tasks_outstanding += 1
    self._send_idx += 1
```

get_data方法比较简单，尝试从data_queue中拿到数据，直到成功，然后返回数据。

```python
def _get_data(self):
    ......
    else:
        while True:
            success, data = self._try_get_data()
            if success:
                return data

```





接着来看主进程真正返回一个batch数据的_next_data方法：

1.  找到第一个合法的rcvd_idx，即该任务所需的数据已经在缓冲区中或者该worker进程任可工作，否则删除该任务因为其已经不可能完成；
2.  若该该任务所需的数据已经在缓冲区中，那么直接返回该数据；
3.  若不在缓冲区中，那么从data_queue中获取数据，如果该数据的idx == rcvd_idx，那么直接返回该数据；若不是，则表明出现乱序，保存在缓冲区中，进行下一次循环。

```python
def _next_data(self):
    while True:
        # If the worker responsible for `self._rcvd_idx` has already ended
        # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
        # we try to advance `self._rcvd_idx` to find the next valid index.
        #
        # This part needs to run in the loop because both the `self._get_data()`
        # call and `_IterableDatasetStopIteration` check below can mark
        # extra worker(s) as dead.
        while self._rcvd_idx < self._send_idx:
            info = self._task_info[self._rcvd_idx]
            worker_id = info[0]
            if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                break
            del self._task_info[self._rcvd_idx]
            self._rcvd_idx += 1
        else:
            # no valid `self._rcvd_idx` is found (i.e., didn't break)
            if not self._persistent_workers:
                self._shutdown_workers()
            raise StopIteration

        # Now `self._rcvd_idx` is the batch index we want to fetch

        # Check if the next sample has already been generated
        if len(self._task_info[self._rcvd_idx]) == 2:
            data = self._task_info.pop(self._rcvd_idx)[1]
            return self._process_data(data)

        assert not self._shutdown and self._tasks_outstanding > 0
        idx, data = self._get_data() # 在这从data queue中拿数据
        self._tasks_outstanding -= 1
        if self._dataset_kind == _DatasetKind.Iterable:
            # Check for _IterableDatasetStopIteration
            if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                if self._persistent_workers:
                    self._workers_status[data.worker_id] = False
                else:
                    self._mark_worker_as_unavailable(data.worker_id)
                self._try_put_index()
                continue

        if idx != self._rcvd_idx:
            # store out-of-order samples
            self._task_info[idx] += (data,)
        else:
            del self._task_info[idx]
            return self._process_data(data)
```

在返回数据时，需要经过_process_data的处理，该函数将revd_idx 加一，并调用try_put_index分配新的任务，接着返回数据。

```python
def _process_data(self, data):
    self._rcvd_idx += 1
    self._try_put_index()
    if isinstance(data, ExceptionWrapper):
        data.reraise()
    return data
```



最后来看看worker进程做了什么，主要是从index_queue中拿到任务，然后通过fetcher.fetch(index)加载数据，最后把数据放入data_queue。大部分代码都是在进行异常处理。

```python
def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                 num_workers, persistent_workers):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    try:
        ......
        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
                
            if isinstance(r, _ResumeIteration): # reset逻辑
                # Acknowledge the main process
                data_queue.put((r, None))
                iteration_end = False
                # Recreate the fetcher for worker-reuse policy # 为什么要重新创建呢？
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                continue
            ......
            
            idx, index = r
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)
                except Exception as e:
                   ......
                   
            data_queue.put((idx, data))
            del data, idx, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    ......
```

