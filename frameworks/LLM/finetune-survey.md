modal-labs llama-finetuning: CodeLlama：看起来直接 pytorch 里的组件就能搞定 7B，一万的数据集，30分钟内训练完10个 epoch

使用 4-bit Transformer 的 QLoRA, [Paper](https://arxiv.org/abs/2305.14314)，[Making LLMs event more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

使用 PEFT(Parameter Efficient Fine Tune): [官方文档](https://huggingface.co/docs/peft/conceptual_guides/lora)。是一个把预训练模型(PLMs)高效适配到下游各种类型程序上而不需要微调所有模型参数的方法。它支持几乎能看到的各类语言模型和图片生成模型

Supervised fine-tuning(SFT):

