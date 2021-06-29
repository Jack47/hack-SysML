Pytorch Data Loader

## 问题
1. 如果数据集有多大没法提前预知怎么办？使用 iterable
2. 实际读取图片和标签数据时，是每个 batch 为一个单位读取嘛？不是，是看 dataset 的实现，比如iterable 可以一次读一个 chunk。而 batch_sampler 给出一个batch的 index后，map风格的 dataset 会读取出这一批。Random DataSet 时呢？其实对于dataloader 只关心要读取的 index list（iterable 要自己去实现 random 啥的，框架不提供）
3. dataloader里多进程(worker_nums> 0)之间如何协作？各自的角色。都是 worker，读取数据，读完的通过 shm ，主进程就能拿到？那怎么合并数据？
4. Pytorch DDP(Distributed Data Parallel) 模式下，多主机的 dataloader 之间如何协作？比如需要 RandomDataSet ： 靠 rank 和 worker_info 来切分数据
5. Dataloader 如何加载非图片数据？比如 21\*21\*21 Tensor。就靠 collate_fn 里转换成 Tensor 了。[data1, data2,...]
6. 读取数据是在 get_item 里读取？此时会不会卡住流程，是否会有预读取？默认 nums_worker = 0，就会卡住。适合某些特定场景
7. forward 里如何消费 batch 的数据？就是一个 tensor，里面细看是第一个维度为 batch 维度的 tensor
8. map style dataset 如何实现？从硬盘读取所有文件，假设是 num_workers = 0。Dataset 里可以自己实现，就跟自己实现 iterator 一样，可以一次性读取出来，一个chun 读取，一个 sample 读取，然后 get_item() 里逐个读出去。
9. dataloader worker 里读取完一个 batch 的数据，是会释放掉的吧，不会傻傻地保存着，，直到一个 epoch 读取完。

## 收获
shuffle=True 时，数据是在每个 epoch 级别做 shuffle的
sampler 和 shuffle 是互斥的
timeout：从 worker 等待一个 batch的超时
prefetch_factor: （默认为2），意味着有总量 2*num_workers 个样本会在整个 workers 里提前获取
persistent_workers：是否在epoch间保留 workers
## Dataset
有两类：map-style 和 iterable-style

map-style：需要支持按照key来检索(`__getitem__()`)数据样本，需要能提前获知数据集长度(`__len__()`)。

iterable-style ：继承自 `IterableDataset`，实现了 `__iter__()` ，代表在数据样本之上的迭代。适合随机读取代价很高甚至不可能的情况，而 batch size 取决于拉取到的数据。比如当在数据集上调用 `iter(dataset)`，会返回一个从数据库或远程 server 上的流式数据。

注意：当在多进程下使用 IterableDataset 处理数据加载时，同样的数据对象在每个worker 进程里复制了很多份，所以需要配置副本来避免获取的数据重复。

## 加载数据的顺序和采样（Sampler）

对于 iterable-style的数据集，数据加载顺序完全由用户定义的迭代器控制。这样可以实现按块读取和动态 batch size（例如每次 yield 出一个batch的样本）。所以 Sampler 对 iterable-style 没意义

对于 map风格的数据集，需要 torch.utils.data.Sampler 类来指定**数据加载时的索引/key的顺序**。它代表了数据集的索引上迭代的对象。例如在 SGD 下，一个 Sampler 会随机排列一个索引列表，每次 **yield 出一个**，或者在 mini-batch SGD 下是 yield 出小规模的数据样本。

DataLoader 里有 shuffle 参数可以决定顺序还是洗牌。也可以使用 Sampler 参数来指定自定义的 Sampler 来每次给出下一次要获取的索引

所以一般用 map 形式的迭代器比较多，比如图片按照 id 来命名。这样需要提前把文件名的全量信息读取到内存里，方便实现map类数据集的两个函数：索引和长度。

## 加载 Batched 和非 Batched  数据
DataLoader 支持自动关联单个的获取数据样本为 batch，通过参数 batch_size, drop_last, 和 batch_sampler

### 自动 Batching （默认）
这是最常见的场景，对应地获取一个 minibatch 数据，并关联为一个batched samples。比如把batch dimesion 作为 Tensor 里的第一个维度。

当 batch_size （**默认为1**）非 None，dataloader 就会返回batced sample，而非普通的单个数据样本。其实就是构造了一个 batch_sampler，每次发出一个keys的列表。

collate_fn 就是把当前batch_sampler 返回的list数据给处理一下：
遍历 batch_sampler
   按照顺序或key的方式获取这一批数据，作为 collate_fn 的输入数组
   
   这个里面可以实现一些 padding 到一个batch里最大长度的逻辑

### 使用 collate_fn
当自动batch关闭：collate_fn 里一次只处理一个单独的数据样本
当自动batch 开启：collate_fn 里一次处理一个list数据。期望它把输入样本关联为一个 batch 之后的数据。接下来描述具体情况：

例如，每个数据样本包含一个三通道的图片和一个整数类别标签，例如数据集的每个元素返回一个二元祖(image, class_index), 默认的 collate_fn 会把这样一堆二元组合并为一个单独的元组：<batch 之后的图片tensor 和一个batch之后的类别索引tensor> 。默认的 collate_fn 有如下的属性：

* 总是把 batch 叠加为新维度
* 自动转换 NumPy 数组和 Python 数值为 Pytorch Tensors
* 保留数据结构，比如每个sample 是字典，就输出一个具有同样key集合的字典，但是把 batched Tensors 作为值（或者是 lists 如果值无法转换为 Tensors）。对于 list，tuple，namedtuple 也是同样

疑问：那上述这里把几个样本合并为一个batch，如果合并key呢？

可以参考 [test_dataloader.py](https://github.com/pytorch/pytorch/blob/510334f34ba1f6ff5c1418f92bfa31c0d7399b6f/test/test_dataloader.py#L1891)

## 单个和多进程数据加载
默认是单进程加载。Python 进程里有全局解释锁(GIL)，阻止了真正实现多线程并发。为了避免数据加载时阻塞计算代码，PyTorch 提供设置 num_workers 来提供多进程数据加载。
### 默认的单个进程加载数据
数据获取是在 DataLoader 倍初始化的同一个进程里执行的。所以可能会阻塞计算。然而，这个模式在如下场景是推荐的：当多进程之间共享数据的资源（比如共享内存，文件描述符）是受限制的，当整个数据集非常小，可以整个读取到内存里。另外单个进程加载提供的出错信息栈更加可读性，debug 起来非常方便。

问题：单进程（同一个进程）里如何一次把数据集都读取出来？除了显性地在 map style dataset 里加载所有的。

### 多进程加载
会以制定数量的单个进程单独加载数据。

每次当 DataLoader 创建时(当调用 enumerate(dataloader)），num_workers 个 worker 进程被创建。此时每个worker里会传入 dataset, collate_fn 和 worker_init_fn 三个函数，用来初始化，获取数据。这意味着数据集的获取在各自内部的 IO，transforms 包括 collate_fn 在worker进程里执行。

如何让每个 worker 负责各自分片里的数据，不打架？torch.utils.data.get_worker_info() 可以获得很多有用信息（包括 worker id，数据集副本，初始种子等）。他在主进程里会返回 None。搜以可以使用这个函数在 dataset 代码或者 worker_init_fn 来划分每个 worker 要负责的分片。上面讲的是对于iterable-style 数据集

因为对于 map 类型的数据集，主进程使用 sampler 来产出索引，发给 worker 来加载数据。所以任何 shuffle 操作在主进程里做。

当迭代过程结束后，worker 会被关闭，或者当 iterator
 被垃圾回收后。
 
警告：通常建议不要在多进程加载数据的场景下返回 CUDA tensors，因为使用 CUDA，在多进程下共享 cuda tensor 有微妙之处。我们推荐使用自动内存固定（设置 pin_memory=True），这样让 CUDA 从 内存里传输数据更快。

### 平台特定的行为

worker 依赖 Python 的multiprocessing，worker 启动的行为在 Windows 和 Unix 航不同：
* Unix 下， fork() 是默认的 multiprocessing 函数。这样子进程可以直接通过克隆的地址空间，访问 dataset 和 Python参数函数。
* Windows 和 MacOS 下，使用 spawn，这个是在主脚本里执行了另外一个解释器，需要一个内部的 worker 函数来通过 pickle 序列化库来接收 dataset，collate_fn。

上述行为是我所不知道的

### 多进程数据加载下的随机性
默认每个 worker 会使用自己的 PyTOrch seed 集合为 base_seed + worker_id，而 base_seed 是一个主进程

## 内存固定(Pinning)
主机到 GPU 的拷贝，当来源于钉住的（page-locked）内存，速度会更快。因为可以用**异步的 GPU copy**。

默认的内存钉住只能识别出Tensor、map和可迭代的 Tensors。custom 类型不行。此时需要自己实现 pin_memory() 接口。




