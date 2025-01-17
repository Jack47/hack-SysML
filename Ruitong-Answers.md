## 两个 Coding 题目

## 1. 处理海量文件

现在在阿里云的对象存储 OSS 上有 4M 个 jsonl 的文件，每个文件大概 100KB。希望你实现：

把这些文件转为 parquet 文件

需要：

> 1. 想出3个以上不同的方案，对比优劣势
>
> 2. 预估你的方案的处理速度，做完需要多久？
>
> 3. 迭代你的方案，直到已经是最快速度了

#### 方案一：利用Python的pandas库进行JSONL到Parquet的转换

利用 Python 的 pandas 库进行 JSONL 到 Parquet 的转换。使用 concurrent.futures 中的 ThreadPoolExecutor 来并行处理多个文件。

```python
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    df = pd.read_json(file_path, lines=True)
    parquet_path = file_path.replace(".jsonl", ".parquet")
    df.to_parquet(parquet_path)
    return parquet_path

def main(input_dir, max_workers=8):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jsonl")]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, files)

main("/path/to/jsonl/files")
```

- 优点：实现简单，易扩展到多线程
- 缺点：单机性能有限，受I/O瓶颈限制
- **预估速度：**假设每个文件的处理耗时为0.1秒，单机8线程理论速度为80万文件/小时，处理4M文件需要约5小时



#### 方案二：使用Apache Spark的pyspark库，将OSS上的数据直接并行读取、转换并写回

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("JSONL-to-Parquet").getOrCreate()

def process_data(input_path, output_path):
    df = spark.read.json(input_path)
    df.write.parquet(output_path)

input_path = "oss://bucket/jsonl-files/"
output_path = "oss://bucket/parquet-files/"
process_data(input_path, output_path)
```

- 优点：可利用多台机器并行处理，适合海量文件
- 缺点：需要配置和管理 Spark集群，初始设置较为复杂
- **预估速度：**假设集群有100个节点，每节点每秒处理500个文件，4M文件可在不到2小时内完成



#### 方案三：使用阿里云Serverless

使用阿里云函数计算（FC），每个函数实例负责处理一部分文件，触发器可以使用OSS事件通知

```python
import oss2
import pandas as pd

def handler(event, context):
    bucket = oss2.Bucket(...)
    file_path = event["file_path"]
    data = bucket.get_object(file_path)
    df = pd.read_json(data, lines=True)
    parquet_data = df.to_parquet()
    bucket.put_object(file_path.replace(".jsonl", ".parquet"), parquet_data)
```

- 优点：无需管理基础设施，弹性扩展
- 缺点：受单实例运行时间限制，需考虑函数冷启动问题
- **预估速度：**4M文件可分配到1万并发实例，每实例处理10个文件，理论上数分钟即可完成



#### 方案四：阿里云DataWorks + EMR

**DataWorks** 提供数据集成和任务调度功能，用于管理海量文件的分布式处理。**阿里云EMR（Elastic MapReduce）** 支持大数据处理，可以直接对OSS文件进行分布式处理。

配置一个阿里云EMR集群，通过OSS批量读取JSONL文件，利用Spark将其转换为Parquet文件，结果存储在OSS中

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("OSS to Parquet") \
    .config("spark.hadoop.fs.oss.endpoint", "oss-cn-hangzhou.aliyuncs.com") \
    .config("spark.hadoop.fs.oss.accessKeyId", "your-access-key") \
    .config("spark.hadoop.fs.oss.accessKeySecret", "your-secret-key") \
    .getOrCreate()

# 读取OSS上的jsonl文件
input_path = "oss://bucket/jsonl-files/"
output_path = "oss://bucket/parquet-files/"

df = spark.read.json(input_path)
df.write.parquet(output_path)
```

- 优点：原生支持阿里云OSS，无需额外中间存储；弹性扩展能力强，可根据文件量动态调整资源，对海量文件处理速度快
- 缺点：依赖阿里云EMR，存在一定使用成本，初始配置稍复杂
- **预估速度：**假设每台机器每秒可处理100个文件，使用100台机器，4M文件可在不到1小时内完成



#### 方案五：OSS批量处理 + Serverless

在阿里云环境中，使用OSS的**Batch Operations（批量操作）**和函数计算（Function Compute）结合。直接在OSS控制台发起任务，这样无需将数据下载到本地。配置触发器，调用函数计算（Function Compute）处理每个文件。使用阿里云SDK或CLI实现自动化：

```python
# 配置批量处理任务
aliyun oss BatchProcess \
    --operation "JsonlToParquet" \
    --input "oss://bucket/jsonl-files/" \
    --output "oss://bucket/parquet-files/"
```

核心处理函数：

```python
import oss2
import pandas as pd

def handler(event, context):
    bucket = oss2.Bucket(...)
    file_path = event["file_path"]
    data = bucket.get_object(file_path)
    df = pd.read_json(data, lines=True)
    parquet_data = df.to_parquet()
    bucket.put_object(file_path.replace(".jsonl", ".parquet"), parquet_data)
```

- **优点：**全程托管，无需维护基础设施。可直接集成到阿里云环境，减少数据迁移时间。

- **缺点：**依赖阿里云OSS和函数计算，跨平台时不可用

- **预估速度：**每个函数实例处理时间约为0.1秒，假设分配10万实例并发，4M文件可在几分钟内完成

  

## 2. 并发推理数据

假设题目1里的 parquet 文件，里面 schema 为 prompt： string

现在需要你实现一个用类似 qwen coder 1.5b 作为 Reward Model 来给这些 prompt 打分。可以参考 https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 。

需要考虑：

> 1. 如何能让这个模型推理速度最快？
>
> - 比如对比 transformer 推理和 vLLM 推理的速度
> - 海量数据要推理，怎么样能让单个推理实例的吞吐最大？
>
> 2. 预估处理上述 4M 个 jsonl 文件，每个里 prompt 文本长度平均为 1k 情况下的处理速度

### A. 如何能让这个模型推理速度最快？

- **动态批处理**: 将不同长度的 prompt 动态分组成等效批次，减少短文本和长文本混合造成 padding 开销，vLLM 自动支持动态批处理
- **数据分片并行化**：将 parquet 文件分片，分发到多台机器或多个 GPU，使用工具 Dask 管理任务分片和调度，每个分片对应一个 GPU 实例，推理完成后合并结果。
- **模型量化**：将模型从 FP16 量化为 INT8，可以使用 bitsandbytes

##### a . transformers推理

```python
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor


def infer_with_transformers(file_path, model_name, output_path, batch_size=32):
    """
    使用 Transformers 推理每个 prompt 的得分，并将结果保存。
    """
    # 加载 tokenizer ，模型和parquet 文件
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
        
    df = pd.read_parquet(file_path)
    prompts = df["prompt"].tolist()

    results = []

    # 动态批量推理
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits[:, 0].tolist()  # 取第一个类别的 logit 分值作为得分

        for prompt, score in zip(batch, scores):
            results.append({"prompt": prompt, "score": score})

    # 保存结果
    output_df = pd.DataFrame(results)
    output_df.to_parquet(output_path)


# 并行处理多个文件
def process_with_transformers(input_dir, model_name, output_dir, batch_size=32):
    """
    并行处理目录中的多个 parquet 文件。
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for file in files:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"scored_{file}")
            executor.submit(infer_with_transformers, input_path, model_name, output_path, batch_size)

if __name__ == "__main__":
    input_dir = "/path/to/parquet_files"
    model_name = "qwen-coder-1.5b" # 模型路径
    output_dir = "/path/to/output_dir"
    process_with_transformers(input_dir, model_name, output_dir)
```

##### b. vLLM 推理

```python
import os
import pandas as pd
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor


def infer_with_vllm(file_path, model_path, output_path, batch_size=32):
    """
    使用 vLLM 推理每个 prompt 的得分，并将结果保存。
    """
    # 加载 parquet 文件并初始化 vLLM 模型
    df = pd.read_parquet(file_path)
    prompts = df["prompt"].tolist()
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=0.0)  # 固定采样参数，确保一致性

    results = []

    # 动态批量推理
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)

        for prompt, output in zip(batch, outputs):
            score = extract_score_from_output(output.text)  # 从生成文本中提取评分
            results.append({"prompt": prompt, "score": score})

    # 保存结果到 parquet 文件
    output_df = pd.DataFrame(results)
    output_df.to_parquet(output_path)


# 自定义评分提取逻辑
def extract_score_from_output(output_text):
    """
    从模型生成的文本中提取评分。
    假设生成结果格式为： "Score: 0.85"
    """
    try:
        score_line = [line for line in output_text.split("\n") if "Score:" in line][0]
        score = float(score_line.split(":")[1].strip())
    except Exception:
        score = 0.0  # 提取失败时返回默认值
    return score


# 并行处理多个文件
def process_with_vllm(input_dir, model_path, output_dir, batch_size=32):
    """
    并行处理目录中的多个 parquet 文件。
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for file in files:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"scored_{file}")
            executor.submit(infer_with_vllm, input_path, model_path, output_path, batch_size)

if __name__ == "__main__":
    input_dir = "/path/to/parquet_files"
    model_path = "qwen-coder-1.5b"  # 模型路径
    output_dir = "/path/to/output_dir"
    process_with_vllm(input_dir, model_path, output_dir)
```

为了使单个推理实例的吞吐最大，可以适当的增大批处理大小，优化序列长度，主要使用动态填充，开启混合精度，优化以后的代码：

```python
import os
import pandas as pd
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor


def infer_with_vllm(file_path, model_path, output_path, batch_size=64, max_length=1024):
    """
    使用 vLLM 推理每个 prompt 的得分，并将结果保存。
    """
    # 加载 parquet 文件并初始化 vLLM 模型
    df = pd.read_parquet(file_path)
    prompts = df["prompt"].tolist()
    llm = LLM(model=model_path, tensor_parallel_size=1, dtype="fp16")  # 加载 vLLM 模型
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_length)

    results = []

    # 动态批量推理
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)

        for prompt, output in zip(batch, outputs):
            score = extract_score_from_output(output.text)  # 从生成文本中提取评分
            results.append({"prompt": prompt, "score": score})

    # 保存结果到 parquet 文件
    output_df = pd.DataFrame(results)
    output_df.to_parquet(output_path)


def extract_score_from_output(output_text):
    """
    从模型生成的文本中提取评分。
    假设生成结果格式为： "Score: 0.85"
    """
    try:
        score_line = [line for line in output_text.split("\n") if "Score:" in line][0]
        score = float(score_line.split(":")[1].strip())
    except Exception:
        score = 0.0  # 提取失败时返回默认值
    return score


def process_with_vllm(input_dir, model_path, output_dir, batch_size=64, max_length=1024, num_workers=8):
    """
    并行处理目录中的多个 parquet 文件。
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file in files:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"scored_{file}")
            executor.submit(
                infer_with_vllm,
                file_path=input_path,
                model_path=model_path,
                output_path=output_path,
                batch_size=batch_size,
                max_length=max_length,
            )

            
if __name__ == "__main__":
    input_dir = "/path/to/parquet_files"
    model_path = "qwen-coder-1.5b"  # 模型路径
    output_dir = "/path/to/output_dir"
    batch_size = 64  # 动态批量大小
    max_length = 1024  # 每条 prompt 最大长度
    num_workers = 8  # 并行处理线程数

    process_with_vllm(input_dir, model_path, output_dir, batch_size, max_length, num_workers)
```



### B. 预估处理速度

假设条件：

1. **数据**

   4M 个 .jsonl 文件，每个文件约 100KB，总数据量 4M * 100KB = 400GB；每条 prompt 平均长度为 1k 字符；假设每个 parquet 文件包含 10,000 条 prompt

2. **模型**

假设同样大小的 qwen coder 1.5b，约 1.5B 参数

3. **硬件配置**

GPU：NVIDIA A100（40GB显存，约 312 TFLOPs FP16 推理性能），单 GPU，支持动态批量

4. **理论推理吞吐量**

$$
\text{推理时间 (T)} = \frac{\text{序列长度 (L)}}{\text{GPU吞吐量 (G)}} \times \text{模型参数 (P)} \times \text{批处理大小 (B)}
$$

1. ##### Transformers

Transformers 框架使用 PyTorch，推理效率主要受限于：序列长度, batch size和模型的大小

假设 312 TFLOPs 性能：

- 单个序列推理时间 ≈ 1ms/token，包括显存 IO 和计算开销
- 每批处理时间 ≈ 32ms，32 条 prompt，每条 1k token
- 每秒吞吐量（TPS）：1 / 32ms ≈ 31 批次/s

单 GPU 每秒处理条数：
$$
31 \, \text{批次/s} \times 32 \, \text{条/batch} = 992 \, \text{条/s}
$$
总推理任务 4M 条：
$$
\text{时间 (T)} = \frac{\text{总条数}}{\text{每秒处理条数}} = \frac{4,000,000}{992} \approx 4032 \, \text{s}
$$
约 **1.12 小时**

2. ##### vLLM

假设每条 prompt 平均 1k token，vLLM 可动态调整到每批处理 64 条，vLLM 的动态批量使吞吐量提升约 30%-50%

- 每批处理时间 ≈ 45ms（64 条 prompt）
- 每秒吞吐量（TPS）：1 / 45ms ≈ 22 批次/s。

单 GPU 每秒处理条数：
$$
22 \, \text{批次/s} \times 64 \, \text{条/batch} = 1408 \, \text{条/s}
$$
总推理任务 4M 条：
$$
\text{时间 (T)} = \frac{\text{总条数}}{\text{每秒处理条数}} = \frac{4,000,000}{1408} \approx 2841 \, \text{s}
$$
约 **0.79 小时**

**进一步优化**，假设有 **8 个线程**，每个线程执行单实例推理：
$$
\text{时间 (T)} = \frac{\text{总条数}}{\text{每秒处理条数}} = \frac{4,000,000}{1408\times8} \approx 355 \, \text{s} = 5.92 \,min
$$
多 GPU 并行化：在 8 x NVIDIA A100 集群下，理论可将总时间减少至 5.92 分钟左右
