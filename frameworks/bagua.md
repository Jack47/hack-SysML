想看如下几个问题：

## 1. hierarichical 部分的代码
好像只能搜到参数，看不到在哪里起作用？

## 2. tcp 优化部分

可借鉴的是  [quick start 和 bencharmk vgg 的部分](https://github.com/BaguaSys/bagua/blob/471daa11fc6547045986c87199bf3083d1e5d746/rust/bagua-net/README.md?plain=1#L11)，有提到 ncc-test 的用法

## 3. flatten parameter?
代码中把这个特性叫做 gradient as bucket view(do flatten)，更能精准表达含义。这个并不能减少那次拷贝，但是会增强程序的局部性。跟 LS 里把所有 param 和 grad 放到一个超大 tensor 里类似的机制

## 4. 第一个 iteration 里体现的 profile 阶段

## 5. fused optimizer
把多个参数的更新，融合到一个或更少的更新里，为了达成这个，用户需要：

1. 把同组的多个参数扁平化(do_flatten=True)
2. perform a fused parameter update by calling fuse_step

这样做好处是减少了参数更新时， kernel 的 overhead

## 6. find_unused_parameters

挂钩子在
```
bagua_build_params
if param.requires_grad 即可找出用户希望计算梯度的，然后给他的 .grad_fn.next_functions[0][0].register_hook()

```
## 7. autotune
```
warmup_times, max_samples
```
看起来主要是输出[这几个参数](https://github.com/BaguaSys/bagua/blob/471daa11fc6547045986c87199bf3083d1e5d746/bagua/service/autotune_system.py#L92)？

```
        "NCCL_MIN_NCHANNELS": 0,
        "NCCL_SOCKET_NTHREADS": 0,
        "NCCL_NSOCKS_PERTHREAD": 0,
        "nccl_buffsize_2p": 0,
```

好像也不是，主要是 recommended buckets
