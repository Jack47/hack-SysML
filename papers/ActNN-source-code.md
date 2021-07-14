## TODO
1. 看看 `enable_quantized_bn`
2. 看看 `cudnn_benchmark_conv2d`
3. 看看 `empty_cache_threshold = 0.2`
4. 看看 `perlayer`
5. `adaptive_bn_scheme`
5. `adaptive_conv_scheme`

## 看 Schema 相关
QBNScheme

config.initial_bits = 8

`Scheme.compute_quantization_bits()`


### Per Sample
如果想用 per-sample 梯度信息，来做自适应量化，需要修改 dataloader 来返回 sample indices，然后配置： `config.use_gradient = True, QScheme.num_samples = 1300000 // size of training set`

### perlayer

怎么体现的每个 layer 级别的 precision？主要想看针对不同 sensity 的东西 => range_sqr * grad

### QScheme 
初始化：

```
QScheme(num_locations=num_locations, group=0, depthwise_groups=1)

```

QScheme 类级别变量：

```
layers[] // 是 QScheme 的实例
num_layers // 
num_samples // 总共一个 epoch 里 samples 的数量
prev_layer // 
batch: dataloader 里 loader i，就设置为 i
```

QScheme.layer 里的变量：
```
.C 
.bits // layer 级别的 bits，是 layer 里N 个sample 的 一个 batch 里 mixed_precision 计算出后(per group的好像)，取个中位数
// 没找到哪里用到 bits 了
```

#### QScheme 函数：
```
// 会在 op.run_backward() 里设置，是 scheme 需要
set_scale(self, grad) // 这个 scale 传递进去的是梯度。里面最终保存的是 norm.square().mean // 此时从向量变数字？
```

### calc\_precision 函数: // 看起来是考虑了梯度(上一次的？) 和 range 之后，对每个数据各自计算出 bit 数

```
calc_precision(initial_bits, Range_sqr * grad, w[1,(N),1], bits * N) 这里计算的公示和目标是什么呢？
b = calc_precision(b, C, w, int(self.bits * N)) // 最后一个是 target，意思是总共的 bit 数？ 这样算出来的 bit 数，就是每个 batch 里的 一个输入 tensor
 有一个了

```
所以精妙之处是即使是 allocate\_perlayer，也可以把对应的 initial\_bits, range\_sqr * grad, 等各自 concat 一下，调用一次 calc\_precision 即可算出
### op.if\_allocate\_perlayer() 
// 在 backward 时调用，而且是只让第一层做： QScheme.allocate_perlayer()
actnn.QBNScheme.allocate\_perlayer() 

```
allocate_perlayer() // 计算出 layers[i].bits
    
```

## 问题
1. 用贪心求解的是要用的 bit 数？它的限定是拿到 sample 了，就能算出来？
2. cuda kernel 里写了 if 条件？是否会影响效率
3. 一些函数的用法，如： atomicOr
4. 哪里是运用不同L的地方
quantization cuda kernel 实现在[quantization_cuda_kernel.cu](https://github.com/ucbrise/actnn/blob/9026b5fe8c3115a326c03a726a92ab87cf176d61/actnn/actnn/cpp_extension/quantization_cuda_kernel.cu?plain=1#L25)

## 看源码的疑问
1. 里面的混合精度 mixed_precision 是什么意思？针对每次 sample 和 每层，**适应性**选取数值精度: config.adaptive_scheme

2. 为何在 convnd 的实现里，在 run\_backward 里需要实现 cudnn\_convolution\_backward 呢，里面还涉及 dilation 矫正。我的理解是不涉及到计算的。并不会把有损量化做纠正。发现这个只是重新链到了 aten 里的实现，不知道为啥要这么搞。这里的参数都是原生 PyTorch 里支持的。可能是因为要保存一些中间值？所以就拦截一下

3. `unpack_single_precision_kernel`
它里面怎么知道边界？比如有多少数字要解压缩出来。看解压本身过程很简单，就是缩放 : 分两种，single_precision 里是固定的 2bit 量化。

看看 mixed precision 怎么做的？其中哪里体现了动态调整 bits 的地方

每个 layer 里用的方法有何不同？

batch normalization 里做了些什么？



## 流程
1. 每次在 run\_forward 之后，把激活值量化一下，保存到 ctx 里
2. 在 run\_backward 需要用到激活值时，反量化出来，然后计算实际算子的 backward 过程

## 目标
### 1. 7.12 看懂 quantize\_activation 里的 pack\_single\_precision_kernel
用 cuda  kernel 算出每个 group 里的 min 和 max，然后计算量化后的数字。这个 group 选择是 256，是跟 warp 里 32 是对应的，要整除效率较高

问题：谁会调用 quantize_activation() ?

`actnn/ops.py`:
```
quantize_activation(input, scheme): // 其他 layer 只是调用这个函数就行吧？所以不应该跟 layer 相关呢
  input_groups, q_bits, q_min, q_max = scheme.compute_quantization_bits(input) // 每个 batch 粒度上算出 bits 数
  q_input, q_scale = quantize_and_pack(input_groups, q_bits, q_min, q_max) // 主要工作量在这里
return q_input, q_bits, q_scale, q_min // 为啥这里只需要 q_min，不需要q_max ？

  
q_input, q_scale = quantize_and_pack(input_groups, q_bits, q_min, q_max)
  // cpp_extension/quantization.cc
  ext_quantization.pack_single/mixed_precision
    pack_single_precision()
      // cpp_extension/quantization_cuda_kernel.cu
      pack_single_precision_cuda()
        compute_scale_single_precision_kernel<<<block, 256>>>(bits, min, max, scale, um_groups) // 有 block 个 warp，每个 warp 里有 256 个 thread ？ => 实际每个 warp 背后 32 个线程承载?
        // 这里就是每 work_per_thread = 8/ bits(2) = 4, 即每个 thread 里会处理4个原始数据，把他们压缩到一个8bit 里
        // val = (data[id] - min[x])*scale + noise
        // 关于 boundary check：有时候输入数据小于实际最终运行的线程覆盖的数据，所以当检查到要处理的数据越界了，就退出
        pack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>()
```

上述方法，对 GPU 非常友好：
1. 每个 group 间可以并发求完 min, max
2. 而之后 per-group 里求每个数据的 quantized 也可以并发，它是每个线程里搞定 pack 之后的一个 8bit 数据的 

2. ActQuantizedMaxPool2d
```
forward()
  act_quantized_max_pool2d_forward_cuda(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    act_quantized_max_pool2d_forward_kernel<<<blocks, threads>>>();
    
```

问题：
1. 为啥要实现 act\_quantized\_max\_pool2d\_forward\_kernel() ? 看里面也没啥特殊的
2. 而 backward\_cuda 里使用的 `AT_DISPATCH_FLOATING_TYPES_AND_HALF`，并没有自己实现


3. ops.py:convd(Function):
```
run_forward(n, forward_op, ctx, input, weight, bias=None, stride=1,)
    quantized = quantize_activation(input, scheme)
    ctx.saved = quantized, weight, bias // 难道是原生 PyTorch 实现里，要保存 input ？
    
    return forward_op(input, weight, bias, stride, padding, dilation, groups)
    // 由于上述 ctx.saved 里没有保存 input，所以这里函数结束，input 值就没了？

run_backward(n, ctx, grad_output)
    ext_backward_func.cudnn_convolution_backward() 
```

问题：
1. `convd.run_forward()` 里， input 和 `quantized_input` 都存在，没看到显示释放 input 的地方呢？
2. ops.conv1d.forward() 里，调用的是 convnd.run_forward(1, F.conv1d, ...)，那什么时候调用 `act_quantized_max_pool2d_forward_kernel` 呢？
3. 原来的 run_backward 是什么样？与现在的差异，对比看才能看出眉目来
4. run_backward() 里为啥要调用自己实现的 `cudnn_convolution_backward()` ?
5. 实际上述 `cudnn_convolution_backward` 是拷贝自 PyTorch 的？

### 2. 7.13 看懂 unpack\_single\_precision

## 基本概念
Warp shuffle: \_shfl\_down_sync

pytorch tensor  slice

block, thread

