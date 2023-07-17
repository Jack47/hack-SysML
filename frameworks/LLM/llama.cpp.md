项目目标是完全用 C/C++ 来推理 LLaMA 模型。实际用到了好几种语言：C(58.2%), C++(18.4%), Cuda(8.6%), Metal(4.0%), Python(3.9%), Objective-C(3.3%)

Apple silicon first-class citizen: 通过 ARM NEON，加速框架和Core ML 来优化：CoreML 里提供了 Apple Neural Engine (ANE)，Metal frameworks

X86 架构：AVX intrinsics 支持（AVX，AVX2，AVX512)

POWER 架构：VSX intrinsics 支持

NVIDIA GPU：cuBLAS

OpenCL GPU：CLBlast

BLAS CPU：OpenBLAS

和 ggml 这个 repo 的关系：示范教育意义的目的，作为 ggml lib 的最主要的试验场，用来开发 ggml 库里的新特性

## Prepare Data & RUN
分两步：

1. 先把模型转换为 ggml fp16 格式
```
python3 convert.py models/7B/
```
2. 再把模型量化到 4-bits (q4_0方法)
```
./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin q4_0
```

在 android 上能跑，有 docker image
## Perplexity(衡量模型质量）
https://huggingface.co/docs/transformers/perplexity
https://thegradient.pub/understanding-evaluation-metrics-for-language-models/
## 兄弟项目 
0. [GGML - 给每个人用的LLM: GG 指作者](): 除了定义了底层 ML 里的原语（比如 tensor类型），它换定义了分发 LLM 的二进制格式
1. [把 OpenAI 的 Whisper 模型(自动语音识别ASR) 使用 C/C++ 实现](https://github.com/ggerganov/whisper.cpp)
