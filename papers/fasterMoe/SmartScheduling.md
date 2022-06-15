# 1. Smart Scheduling提出的背景

​	在分布式训练中，一般的模型训练分为两种：数据并行和模型并行。	

​	在GShard论文中，针对MoE提出一种专家并行的方法：即将不同的专家(MLP)放置在不同的GPU上，并行操作以加快模型的训练(如下图)。在图中，不同woker将token(训练数据)通过All-to-all操作来传输给期望的专家，在得到输入之后，专家进行计算之后得到结果，再将结果进行All-to-all操作传送回对应的Worker之中，用于作为下一层的输入。

<img src="https://laekov.com.cn/l/figures/fastermoe/expertpara.png" alt="img" style="zoom: 50%;" />

​	在这个过程之中，存在一个问题：集合通信算子效率较低，即要等整个 all-to-all 操作发完所有数据之后才能继续进行操作, 对于计算和通信的硬件资源都有比较大的浪费。

​	从下图我们可以发现，当我们在进行通信操作之时，我们的计算硬件是空闲的，在进行计算操作的时候，通信硬件是空闲的。专家并行之中的all-to-all操作借助于NCCL库中函数进行实现，当我们想对通信和计算这一套粗粒度的流程进行分割之时，由于不同的通信和计算任务存在很强的依赖关系，如果数据传输的顺序设置不当很容易造成死锁。

​													![img](http://tiebapic.baidu.com/forum/w%3D580/sign=c79e16d4639759ee4a5060c382fa434e/20d71e12b31bb051377dc1f1737adab44bede02b.jpg?tbpicau=2022-06-17-05_73132500f418f997eec2442640503c2c)

# 2. Smart Scheduling的优化策略

​	对于Gshard中，集合通信算子效率较低的问题(操作同步执行模型效率低下)。在FasterMoE中引入了异步细粒度智能调度策略(Asynchronous Fine-Grained Smart Scheduling)，将任务划分为更小的部分来进行，并在考虑效率的情况下重新调整细粒度的通信和计算操作。细粒度的通信和计算操作允许异步执行计算和通信操作，从而更好的利用硬件。

​	在优化的过程中，我先首先将worker进行分组来进行all-to-all操作，并使用Pair-wise Exchange算法来执行all-to-all操作。

**Pair-wise Exchange算法**：

​	先将worker分成大小拆分成n组，n个组构成一个环(这里感觉很像Ring All-Reduce算法)，并以从0到n-1的步幅将数据发送给别的组。采用启发式的方法，将紧密联系的worker放在同一组之中，使得在同一组之中的worker联系的更加快速。而组的大小则是由连接拓扑和计算粒度来决定。

​								<img src="https://tiebapic.baidu.com/forum/pic/item/3abb9751f8198618f936935d0fed2e738ad4e613.jpg?tbpicau=2022-06-17-05_60934e3995711018126d66991f6b17a3" alt="img" style="zoom:50%;" />

​	如上图所示，$W_i$代表了不同的worker，有$n$个worker便对应$n$步，其中第$i$步，$W_i$将token发送到$W_{(j-i)\,mod\,n}$，接受来自$W_{(j+i)\,mod\,n}$的数据。经过$n$轮，每个worker从其他的worker上接受了数据，也将数据发给了其余的worker，这样一次all-to-all操作便被完成。

**粗粒度的划分**：

​	具体的执行过程，参考了作者的Slides：

​	我们将粗粒度的操作划分为三个细粒度的操作，分别为$S,C,R$，下面有一个例子来说明

​	首先，假定我们有三个worker，以worker1为例子：

- $S_i$：Worker $i$ 发送输入给worker1（第一次all-to-all操作的一部分）

  ![img](http://tiebapic.baidu.com/forum/w%3D580/sign=3b9b59d11643fbf2c52ca62b807cca1e/cab64a224f4a20a4bbe4401ed5529822700ed096.jpg?tbpicau=2022-06-17-05_c5743ace44a157b6edb41652f31030c6)

- $C_i$：在worker1上的expert1对来自worker $i$进行计算

![img](http://tiebapic.baidu.com/forum/w%3D580/sign=c9ed8fb5553853438ccf8729a311b01f/3b5b64166d224f4adc630fca4cf790529a22d196.jpg?tbpicau=2022-06-17-05_640c512948c65a12bd821221af37319b)

- $R_i$：Worker1将计算得到的output输出给Worker $i$(第二次all-to-all操作的一部分)

​											![img](http://tiebapic.baidu.com/forum/w%3D580/sign=eeff1aaabef2b211e42e8546fa826511/e482684a20a44623f730d9bbdd22720e0ef3d796.jpg?tbpicau=2022-06-17-05_2341aaeeb8448fee4c51d7658b75b2e6)



**细粒度操作的排列：**

​	在上述过程之中，三个过程之中，每一组三个操作之间有数据依赖关系，而其余的操作没有数据依赖关系，即可以同时执行。

我们可以遵循数据依赖关系，将$C$和$R$尽快的执行，可以得到一个低延迟(这里R1开始在C2执行偏后位置，是因为通信设备被S3操作占用)：

​											![img](http://tiebapic.baidu.com/forum/w%3D580/sign=f7f927ef58950a7b75354ecc3ad3625c/8aef5ff0f736afc3c2c043bcf619ebc4b54512bf.jpg?tbpicau=2022-06-17-05_2f3446fa6a29c96606e5c6ea0f911534)

为了进一步最小化延迟，论文中给出了以下解释：

- 使用一组worker来替代一个worker
- 使用启发式的算法，最小化第一个$S$操作和最后一个$R$来进一步降低延迟(实现没太看懂)

![img](http://tiebapic.baidu.com/forum/w%3D580/sign=13f005e12b061d957d4637304bf60a5d/22c4244f78f0f736153230d84f55b319e9c413bf.jpg?tbpicau=2022-06-17-05_a76b94647c283feac68ccaa08da619ae)

# 3. 源码解读

**smart_schedule.h**

```c++
// 借助nccl库对需要all-to-all的信息进行发送和接受
template<typename scalar_t>
void exchangeWith(
        const scalar_t* sendbuf, size_t sendcount, int t_send,
        scalar_t* recvbuf, size_t recvcount, int t_recv,
        long d_model,
        cudaStream_t stream, ncclComm_t comm) {
    if (sendcount) {
        ncclSend(sendbuf, sendcount * d_model * sizeof(scalar_t),
                ncclChar, t_send , comm, stream);
    }
    if (recvcount) {
        ncclRecv(recvbuf, recvcount * d_model * sizeof(scalar_t),
                ncclChar, t_recv, comm, stream);
    }
}

/*
 简单的宏定义，用来计算本次S操作需要发送到哪个worker和介绍哪个worker的发送
*/
#define GEN_BASE(_step) \
    long to_base = (group_rank + _step) % n_groups * pipeline_gran; \
    long from_base = (group_rank + n_groups - _step) % n_groups * pipeline_gran;

/*
 简单的宏定义，用来计算需要发送
*/
#define GEN_IDX \
    int idx_send = ei + rank_send * num_expert; \
    int idx_recv = ei + rank_recv * num_expert; \
    int gidx_send = ei * world_size + rank_send; \
    int gidx_recv = ei * world_size + rank_recv; \
    int idx_self = ei +      rank * num_expert;

/*
功能：计算local_ptr、global_ptr、local_global_ptr

num_expert: 每个worker的专家数量，一般默认为1
rank: 没有搞懂什么意思
world_size: 含有专家的woker数量(一般是GPU数量)
local_expert_count: 本worker发出的feature中到每个expert的数量
global_expert_count: 本worker收到的feature中来自每个worker的数量
stored_models: 看模型参数是否存储,影子专家
local_ptr: 本worker发出的feature到每个expert数量的累计和
loacl_global_ptr和global_ptr: 分别是local_expert_count和global_expert_count的前缀和，表示本地的数组里每个其它worker对应的数据的偏移量

*/
void computePtrs(long num_expert, long rank, long world_size,
        const long* local_expert_count,
        const long* global_expert_count,
        const bool* stored_models,
        int *local_ptr,
        int *global_ptr,
        int *local_global_ptr) {
    local_ptr[0] = global_ptr[0] = local_global_ptr[0] = 0;
    // 遍历所有的worker,计算local_ptr,local_global_ptr
    for (int i = 0; i < num_expert * world_size; ++i) {
        local_ptr[i + 1] = local_ptr[i] + local_expert_count[i];

        local_global_ptr[i + 1] = local_global_ptr[i];
        // if model fetched, add local tokens
        if (stored_models[i]){
            local_global_ptr[i + 1] += local_expert_count[i];
        }

        // 计算worker的组号和组中序号
        auto expert_idx = i % num_expert;
        auto worker_idx = i / num_expert;

        // 用来表示全局的坐标
        auto gp_idx = expert_idx * world_size + worker_idx;
        // if local model wasn't fetched, receive global tokens
        if (stored_models[rank * num_expert + expert_idx]) {
            global_ptr[gp_idx + 1] = 0;
        } else {
            global_ptr[gp_idx + 1] = global_expert_count[i];
        }
    }
    global_ptr[0] = 0;
    for (int i = 0; i < num_expert * world_size; ++i) {
        global_ptr[i + 1] += global_ptr[i];
    }
}

/*
    计算模板，根据专家模型的架构利用GPU进行计算，将inp_buf中内容输入专家，进行计算输出到out_buf
*/

template<typename scalar_t>
void computeFn(py::function fn, c10::Device device,
        scalar_t* inp_buf, scalar_t* out_buf,
        long idx, long offset, long micro_batch_size, long d_model,
        CudaStreamManager* smgr) {
    if(micro_batch_size == 0) {
        return;
    }
    auto options = torch::TensorOptions()
        .dtype(c10::CppTypeToScalarType<scalar_t>::value)
        .device(device)
        .requires_grad(true);
    auto inp = torch::from_blob(inp_buf + offset * d_model,
            {micro_batch_size, d_model}, options);
    auto oup = torch::from_blob(out_buf + offset * d_model,
            {micro_batch_size, d_model}, options);
    smgr->use_default = true;
    fn(inp, oup, idx);
    smgr->use_default = false;
}

/*
	利用CUDA的正向计算过程，涉及到S、C、R操作
*/

template<typename scalar_t>
void fmoe_cuda_fused_forward_impl(
        py::function forward_fn,
        py::function stash_fn,
        py::function pop_fn,
        c10::Device device,
        std::vector<torch::Tensor> params,

        scalar_t* input_buf,
        scalar_t* global_input_buf,
        scalar_t* global_output_buf,
        scalar_t* output_buf,

        const long* local_expert_count,
        const long* global_expert_count,
        const bool* stored_models,

        long d_model,
        long num_expert, long rank, long world_size, long expert_size,
        long pipeline_gran, CudaStreamManager* smgr) {
    auto torch_stream = c10::cuda::getCurrentCUDAStream().stream();

    int *local_ptr = new int[num_expert * world_size + 1];
    int *global_ptr = new int[num_expert * world_size + 1];
    int *local_global_ptr = new int[num_expert * world_size + 1]; // local fetched models tracker
    computePtrs(num_expert, rank, world_size,
            local_expert_count, global_expert_count, stored_models,
            local_ptr, global_ptr, local_global_ptr);

    if (pipeline_gran > world_size) {
        pipeline_gran = world_size;
    }
    // n_groups: 分组的个数
    // group_rank: 代表了在第几组
    long n_groups = world_size / pipeline_gran;
    long group_rank = rank / pipeline_gran;

    cudaEvent_t *input_ready = new cudaEvent_t[n_groups];
    cudaEvent_t *output_ready = new cudaEvent_t[n_groups];
    for (long i = 0; i < n_groups; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
    }

    // S_0 ... S_n(本模块是进行S操作，对数据进行发送)
    for (long step = 0; step < n_groups; ++step) {
        for (long ei = 0; ei < num_expert; ++ei) {
            // 宏替换，计算发送到的组和接受哪一组的token
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            // 每个组都有pipeline_gran个worker
            for (int j = 0; j < pipeline_gran; ++j) {
                // rank_send: 指定token发送给哪一个worker
                int rank_send = j + to_base;
                // rank_recv: 用来定位本worker接收哪一个worker发来的token
                int rank_recv = j + from_base;
                
                GEN_IDX;
                
                // 进行数据的发送
                exchangeWith(input_buf + local_ptr[idx_send] * d_model,
                        local_expert_count[idx_send] * !stored_models[idx_send], rank_send,
                        global_input_buf + global_ptr[gidx_recv] * d_model,
                        global_expert_count[idx_recv] * !stored_models[idx_self], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
        cudaEventRecord(input_ready[step], smgr->stream(0));
    }

    // Broadcast shadowed experts(将影子专家进行广播，这是论文中对负载不均衡的改进)
    cudaEvent_t evt_get, *evt_shadow;
    if (params.size() > 0) {
        evt_shadow = new cudaEvent_t[params.size()];
    }
    for (long i = 0, si = 0; i < world_size * num_expert; ++i) {
        if (stored_models[i]) {
            if (i / num_expert == rank) {
                cudaEventCreate(&evt_get);
                cudaEventRecord(evt_get, torch_stream);
                cudaStreamWaitEvent(smgr->stream(1), evt_get);
                cudaEventDestroy(evt_get);
            }
            NCCL_SAFE_CALL(ncclBcast((void*)params[si].data_ptr<scalar_t>(),
                        expert_size * sizeof(scalar_t), ncclChar,
                        i / num_expert, smgr->ncclcomm, smgr->stream(0)));
            cudaEventCreate(evt_shadow + si);
            cudaEventRecord(evt_shadow[si], smgr->stream(0));
            ++si;
        }
    }

    // C_0 ... C_n(各个Experts收到input，进行正向计算)
    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(torch_stream, input_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            long offset = global_ptr[ei * world_size + from_base];
            long micro_batch_size = global_ptr[ei * world_size +
                (from_base + pipeline_gran)] - offset;
            computeFn(forward_fn, device,
                    global_input_buf, global_output_buf,
                    step, offset, micro_batch_size, d_model, smgr);
        }
        cudaEventRecord(output_ready[step], torch_stream);
    }

    // Compute over shadowed experts(对负载不均衡情况，利用影子专家进行计算)
    for (long i = 0, si = 0; i < world_size * num_expert; ++i) {
        if (stored_models[i]) {
            stash_fn(params[si], si);
            cudaStreamWaitEvent(torch_stream, evt_shadow[si], 0);
            long offset = local_ptr[i];
            long micro_batch_size = local_expert_count[i];
            computeFn(forward_fn, device,
                    input_buf, output_buf,
                    n_groups + si, offset, micro_batch_size, d_model, smgr);
            ++si;
        }
    }
    pop_fn();

    // R_0 ... R_n (对计算的结果再进行一次all-to-all操作，讲输入的token进行计算后，传播到)
    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(smgr->stream(0), output_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + from_base;
                int rank_recv = j + to_base;
                GEN_IDX;
                exchangeWith(global_output_buf + global_ptr[gidx_send] * d_model,
                        global_expert_count[idx_send] * !stored_models[idx_self], rank_send,
                        output_buf + local_ptr[idx_recv] * d_model,
                        local_expert_count[idx_recv] * !stored_models[idx_recv], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
    }

    delete [] local_ptr;
    delete [] global_ptr;
    delete [] local_global_ptr;
    checkCudaErrors(cudaGetLastError());
    for (long i = 0; i < n_groups; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
    }
    for (unsigned i = 0; i < params.size(); ++i) {
        cudaEventDestroy(evt_shadow[i]);
    }
    delete [] input_ready;
    delete [] output_ready;
}






```



**其他两个源码的注释解读，写的比较乱，等周末没课了，再进行总结**



# 4. 有关疑问点(补充说明)

**说明**：

- num_expert必须设为1
- 专家的输入和输出的向量长度必须相等

**疑问**：

1.其中NCCL库的有关函数看不太懂

2.上述注释157行的ei没太明白啥意思，是预留一个单worker多GPU情况？
