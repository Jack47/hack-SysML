
看实现：

## 概念
  // When we are inside a vmap, all tensors dispatch on this key.
  // See Note: [DispatchKey::VmapMode usage] for more details.
  
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



## DTR 和 Capuchin 的对比
噢？capuchin 里，recompute 是二等公民，swap因为用的 PCIe，并不跟 GPU 并发冲突，所以优先 swap 的
可能不能recompute 提前是因为实现复杂点，目前  pytorch 里默认用一个 stream，所以多个不同类型 kernel 是串行的，不会并发
而且显存用很多时，计算也达到峰值了，没有空间用来并发 recompute（记得论文里这么写的
## 参考资料
1. [issues met when implement Dynamic tensor rematerialization](https://github.com/pytorch/pytorch/issues/62448): pytorch 的作者介绍了 一些对 DTR 第一版的一些问题和看法
2. [Reimplementing DTR in generic](https://github.com/uwsampl/pytorch/pull/62)
3. [first dtr pr](https://github.com/pytorch/pytorch/pull/42056)
