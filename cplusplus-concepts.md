## std::move
Eliminating spurious copies. 

## reinterpret_cast
如文字的意思，主要用于重新解释指针、引用这两类数据类型。比如 lightseq 里实现的I/O用 fp16，计算用 fp32，就是在kernel 里通过 reinterpret_cast 来解释:

out/int 都是 fp16的数组指针,但是里面计算时转化成了 float

```
__global__ void ls_droppath_bwd_kernel(const int total_count, const float drop_prob,
                                      __half *out, const __half *in,
                                      const __half *rand, const int channels) {
  float4 *out4 = reinterpret_cast<float4 *>(out);
  const float4 *vals_float4 = reinterpret_cast<const float4 *>(in);
```

```
reinterpret_cast<new_type>(expression)
```

Returns a value of type new\_type


```
const float4 *input4 = reinterpret_cast<const float4 *>(input);
```
