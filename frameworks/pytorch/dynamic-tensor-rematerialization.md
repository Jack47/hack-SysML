## 用法
在第一个输入那里，把 input 转化为 Checkpoint 类型的 tensor，它后续通过计算图出来的所有中间状态，都会是 CheckpointTensor 类型的，都会交给 CheckpointTensor 相关重载的函数去处理

```
x = torch.Tensor([1]).checkpoint()
y = x
z = y
```
## 实现
### E1. 集成到 PyTorch：

为了避免修改 PyTorch 太多代码，实现是包装了 tensor 的实现。即，在 PyTorch 里添加了 tensor 的一个实现：CheckpointTensor，它在 PyTorch 已有实现上，添加了跟踪父节点操作和其他元数据（上次访问时间）和父操作耗时（这些都是第一次创建tensor时得到的耗时）。然后把 tensor 注册到 DTR 运行时。算子的耗时计算用了系统时间，为了保证计算正确，我们强制PyTorch进入同步执行模式（让 GPU 算子顺序执行）；发现 DTR在不改为同步执行模式下，也能大大减少显存预算，尽管这个会破坏 DTTR 记录算子的耗时。

evictions 时，CheckpointTensor 可以释放底层的 tensor 显存；它保存了重计算父算子的闭包（如何做到？），这样当需要这个 tensor 时，就可以让运行时重计算。为了处理原始程序的释放操作，也会汇报 DTR 运行时 对引用计数的增加和减少。checkpoint() 函数可以让普通 tensor 变成 CheckpointTensor，decheckpoint 可以做反向操作（loss 和 output 最终需要这样做）

我们修改后的 PyTorch 版本会把任何涉及 CheckpointTensor 的算子分发给特定的 CheckpointTensor 实现。这个机制就是 PyTorch 提供的，例如把 GPU 管理的 ttensor 分发给 CUDA的实现。CheckpointTensor 的重载版本实现会在调用已有实现的基础上，把结果包装到 CheckpointTensor 里。因为 PyTorch 里所有tensor 访问都是通过算子做的，所以更新原始信息如访问时间等只需要在 CheckpointTensor 的对应重载的算子里调用 DTR 的运行时即可。

DTR 运行时是一个简单的单例，保存了一个所有 CheckpointTensor 的pool。同时负责维护启发算法所需要的类数据结构（当一个 CheckpointTensor被驱逐或者重计算时更新）。在每个 CT 操作之前，DTR运行时检查是否显存预算超了。如果是，就检查pool里的 CT，计算启发分数，然后把得分最少的驱逐掉，直到不能驱逐或者预算够了（当前的prototype实现容许分配一个tensor 后超预算。可以通过给 PyTorch的GPU内存管理里添加一个回掉，让分配请求时就调用 DTR 运行时。为了简单就没实现上述方法）。可以优化启发算法的开销，比如排好序。DTR 运行时也负责实现 json格式的事件(算子，引用计数)log 机制

DTR prototype 支持 PyTorch 实现细节如 in-place 修改，aliasing, multiple operator outputs.DTR 通过引入copy-on-write修改层来支持in-place 修改: 天天通过把来源 tensor 拷贝之后，再修改拷贝，就可以让算子是纯函数。（类似地，batchnorm和dropout等非纯函数，可以把 PRNG 种子当作输入的一部分传入，把修改传递出去）。DTR 运行时会把这些 CT 的拷贝算子重载到修改算子上。为了支持结果是输入参数别名的算子，DTR 运行时把所有这种别名 CT 放到了单独的 alias pools 里。当一个 alias pool 里的元素被驱逐，所有在 alias pool里对应的元素都当作被驱逐。但是 alias 是再次需要时，单独重计算的。对于多个输出算子产出的 CT，DTR 允许他们被单独驱逐，但保证会一起重计算出来

### E.2 运行时优化

## 概念
  // When we are inside a vmap, all tensors dispatch on this key.
  // See Note: [DispatchKey::VmapMode usage] for more details.
  
## Corner Cases:
1. CT tensor如果被计算多次，那只想计算一次梯度。所以wrapper必须是 require_grad, 而wrapped value 不需要
2. 常量 CT 是不可 evictable的
3. op 返回多个输出的情况：rematerializer 在之间共享。所以执行一次，就拿到所有的结果
4. op可能是inplace的情况：不返回输入，只修改输入。此时 COW operator,用 ref 来包装 CT。内部的 CT 就可以纯函数
5. 可能修改 uncheckpointed tensor，不支持，报错
6. create aliases：使用 AliasPool 来跟踪，每个 AliasPool 保存一个 tensor 的集合，是互为 alias 的关系。
### 7. op 可能会创建一个对不可 evict tensor 的 Alias。
不支持这种情况，会报错：如果tensor有任何live Alias。为啥不支持？

要如何支持？

然而可以：每个 AP 保存 External Reference 的 weak pointers。当 alias 修改发生，会使用 rematerialize_function 来在 base tensor (other tensor alias from)，然后输出所有新的alias 的值，最后更新 Ref

## Memory Safety:
对象会有很多 backedges. 为了收集计算完成后的内存信息，需要所有 strong pointer 是如下形式： value -> input。确保每个 external ref 之后，就可以释放

## 优化：
把没有外部 ref 的 tensor 这样对待：

认为他的下次使用时间是无限大，所以 evict 的时候，会立马 evict

### Boxed functions vs C++ calling convention
a boxed kernel function with signature `void(const OperatorHandle&, Stack*)`. i.e., they receive a stack of arguments in a boxed calling convention, ranther than in the native C++ calling convention. Boxed functions are typically only used to register backend fallbacks via torch::Library::fallback().

### Boxed Fallback/Backend fallbacks

为什么有这个东西？看起来是8月份新出的，主要是更优雅的实现，以前需要手工写/codegen 里来实现一些操作，现在可以用 Boxed Fallback 处理一类 OP

Register a **fallback** implementation for all operators which will be used if there is not a specific implementation for an operator available. There MUST be a **DispatchKey** associated with a fallback; e.g., only call this from TORCH\_LIBRARY\_IMPL() with namespace `_`. Unboxed functions typically do not work as fallback functions, as fallback functions must work for every operator(even though they have varing type signatures)

fallback 举例： 下面的 TESTING_ONLY_GenericMode 是 DispatchKeySet

```
 auto gm = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericMode); // 这个和 L107 行的声明有啥区别？
  gm.fallback(torch::CppFunction::makeFallthrough());
```

```
  auto m = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericWrapper);
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&generic_wrapper_fallback>());

```

这种也算 fallback 的一种？只不过不是挂在某类实现上，而是更细粒度，具体到某类操作：
```
  auto m = MAKE_TORCH_LIBRARY_IMPL(aten, TESTING_ONLY_GenericMode);
  m.impl("mul.Tensor", torch::CppFunction::makeFromBoxedFunction<&generic_mode_fallback>());
```
## Pytorch 里用法
```
  void generic_mode_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {}
  m.impl("mul.Tensor", torch::CppFunction::makeFromBoxedFunction<&generic_mode_fallback>());
```

## 问题
0. 为什么 DTR 不要求静态图？
1. 如何记录一个 tensor 的创造过程？比如在 pytorch 哪个地方记录 tensor 访问时间，父操作节点呢？
2. 如何决定 rematerialization 时机
3. 与 Captuchin 区别和联系

## 实现
主要有这么几块：

1. [CheckpointTensorImpl.h](https://github.com/uwsampl/dtr-prototype/blob/eff53cc4804cc7d6246a6e5086861ce2b846f62b/dtr_code/dtr-implementation.patch#L762-L812) : 所有这些实现的 tensor 都可以被清理/重计算，包括一个 checkpoint dispatch key
2. [Checkpoint.cpp](https://github.com/uwsampl/dtr-prototype/blob/eff53cc4804cc7d6246a6e5086861ce2b846f62b/dtr_code/dtr-implementation.patch#L1505): Tensor checkpoint_add() 之类，实现每个 op 的 rematerialize 操作
3. [New dispatcher functions for some backward functions(是说为了向前兼容？). native_functions.yaml](https://github.com/uwsampl/dtr-prototype/blob/eff53cc4804cc7d6246a6e5086861ce2b846f62b/dtr_code/dtr-implementation.patch#L3944)
4. [Some generic PyTorch fixes]()
## 线索

1. The best description of the implementation aspects of this code are in Appendix E
2. boxing/unboxing fallback => 免去人工 codegen ？ `aten/src/ATen/core/dispatch/backend_fallback_test.cpp` , vmap fallback kernel

## 参考资料
1. [issues met when implement Dynamic tensor rematerialization](https://github.com/pytorch/pytorch/issues/62448): pytorch 的作者介绍了 一些对 DTR 第一版的一些问题和看法
2. [Reimplementing DTR in generic](https://github.com/uwsampl/pytorch/pull/62)
3. [first dtr pr](https://github.com/pytorch/pytorch/pull/42056)
4. [DTR 作者知乎上的通俗解释](https://www.zhihu.com/people/marisa.moe/answers)
