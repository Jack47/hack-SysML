## Step 1: Reducing cpu overhead through torch.compile and a static kv-cache (107 TOK/s)
```
torch.compile(decode_one_token, model='reduce-overhead', fullgraph=True)
```

显存带宽利用率：

```
MBU = #Params * bytes per param * tokens per second / memory-bandwidth
```
比如 7B 的模型，使用 fp16 存储参数，达到了 107 tokens/s。最终，我们的 A100-80GB 有理论上 2 TB/s 的显存带宽。那么上述算下来是 72%

ML-LLM 在各种异构设备的 4-bit 量化上性能很好
