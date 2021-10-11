
先要通过 `cuda_device = torch.device('cuda')` 或 `torch.device('cuda:0')` 来指定当前使用哪个 GPU 设备

然后在创建 tensor 时指定所在的设备: `x = x.to(cuda_device)`

## 其他 api

### torch.cuda.memory_summary
输出举例：
```

```
