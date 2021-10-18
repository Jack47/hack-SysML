
先要通过 `cuda_device = torch.device('cuda')` 或 `torch.device('cuda:0')` 来指定当前使用哪个 GPU 设备

然后在创建 tensor 时指定所在的设备: `x = x.to(cuda_device)`

## 其他 api

### torch.cuda.memory_summary
输出举例：
```

```

### torch.cuda.memory_stats(devices=None)

输出举例：
```

```

### torch.tensor.detach()
返回一个新的 Tensor 对象，并且新 Tensor 是与当前的计算图分离的，其 requires_grad=False，反向传播时不参与计算梯度。而且与被 detach的tensor共享数据部分的存储空间

### torch.tensor.requires_grad_()
告知需要梯度

CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));

疑问：默认不设置上述语句，此时难道是多个kernel 同时执行的？
