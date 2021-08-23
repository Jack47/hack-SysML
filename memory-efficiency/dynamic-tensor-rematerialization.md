
看实现：

## 概念
  // When we are inside a vmap, all tensors dispatch on this key.
  // See Note: [DispatchKey::VmapMode usage] for more details.
  
### Boxed functions vs C++ calling convention
a boxed kernel function with signature `void(const OperatorHandle&, Stack*)`. i.e., they receive a stack of arguments in a boxed calling convention, ranther than in the native C++ calling convention. Boxed functions are typically only used to register backend fallbacks via torch::Library::fallback().

### Fallback

Register a **fallback** implementation for all operators which will be used if there is not a specific implementation for an operator available. There MUST be a **DispatchKey** associated with a fallback; e.g., only call this from TORCH\_LIBRARY\_IMPL() with namespace `_`. Unboxed functions typically do not work as fallback functions, as fallback functions must work for every operator(even though they have varing type signatures)

fallback 举例：

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

## 线索

1. The best description of the implementation aspects of this code are in Appendix E
2. boxing/unboxing fallback => 免去人工 codegen ？ `aten/src/ATen/core/dispatch/backend_fallback_test.cpp` , vmap fallback kernel

