# 1.处理海量文件

 

现在在阿里云的对象存储 OSS 上有 4M 个 jsonl 的文件，每个文件大概 100KB。希望你实现：

把这些文件转为 parquet 文件

需要：

1.想出3个以上不同的方案，对比优劣势

2.预估你的方案的处理速度，做完需要多久？

3.迭代你的方案，直到已经是最快速度了



## 方案1:单机计算

### 单机单线程

**代码实现**

```python
# 本地临时目录
temp_dir = '/tmp/oss_processing'

# 下载文件列表
file_list = ['path/to/file1.jsonl', 'path/to/file2.jsonl', 'path/to/file3.jsonl']

def jsonl_to_parquet(jsonl_file):
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    df = pd.DataFrame(data)
    
    table = pa.Table.from_pandas(df)
    parquet_file = jsonl_file.replace('.jsonl', '.parquet')
    
    pq.write_table(table, parquet_file)
    
    return parquet_file

def process_files():    
    for oss_key in file_list:
        local_jsonl_path = os.path.join(temp_dir, os.path.basename(oss_key))
        
        download_file(oss_key, local_jsonl_path)
        parquet_file = jsonl_to_parquet(local_jsonl_path)
        upload_file(parquet_file, oss_key.replace('.jsonl', '.parquet'))
```

性能瓶颈分析：

- 磁盘I/O：4M个文件总大小：4M * 100KB = 400GB，如果内存大小不足需要将文件内容写入磁盘，频繁的磁盘I/O在处理大量文件时会产生较大影响。
- CPU瓶颈：虽然单个文件转换时间较短但是面对大规模数据，整体处理时间会很长，单线程不能充分利用CPU资源

优化策略：

- 多线程优化I/O操作：多线程下载/上传文件
- 进程池优化文件转换：多进程处理文件转换操作



### 单机多进程

**代码实现**

```python
# 本地临时目录
temp_dir = '/tmp/oss_processing'

# 下载文件列表
file_list = ['path/to/file1.jsonl', 'path/to/file2.jsonl', 'path/to/file3.jsonl']

def jsonl_to_parquet(jsonl_file):
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    df = pd.DataFrame(data)
    
    table = pa.Table.from_pandas(df)
    parquet_file = jsonl_file.replace('.jsonl', '.parquet')
    
    pq.write_table(table, parquet_file)
    
    return parquet_file

def process_file(oss_key):
    local_jsonl_path = os.path.join(temp_dir, os.path.basename(oss_key))
    
    download_file(oss_key, local_jsonl_path)
    parquet_file = jsonl_to_parquet(local_jsonl_path)
    upload_file(parquet_file, oss_key.replace('.jsonl', '.parquet'))
    
def process_files_parallel():
    start_time = time.time()
    
    # 使用multiprocessing创建进程池并并行处理文件
    with Pool(processes=8) as pool:
        pool.map(process_file, file_list)
```

一次最多启动多个进程，加速文件转换计算过程

**时间估算：**

假设开启8个进程，每个文件处理时间100ms：4M / 8 = 13.8h

**性能瓶颈**：磁盘I/O，由于是单机处理，多个进程并发访问磁盘，磁盘的带宽可能会被耗尽，导致磁盘 I/O 的延迟和性能下降。



## 方案2: 使用 Hadoop MapReduce

**伪代码**

```java
public class JSONLToParquetMapper extends Mapper<Object, Text, NullWritable, GenericRecord> {

    private static final Schema SCHEMA = new Schema.Parser().parse(new File("schema.avsc"));
    private AvroParquetWriter<GenericRecord> writer;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        Path outputPath = new Path("output.parquet");
        writer = new AvroParquetWriter<>(outputPath, SCHEMA);
    }

    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        JSONObject json = new JSONObject(value.toString());

        GenericRecord record = new GenericData.Record(SCHEMA);
        record.put("field1", json.getString("field1"));
        record.put("field2", json.getInt("field2"));

        writer.write(record);
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        if (writer != null) {
            writer.close();
        }
    }
}

public class JSONLToParquetReducer extends Reducer<NullWritable, GenericRecord, NullWritable, GenericRecord> {
    @Override
    protected void reduce(NullWritable key, Iterable<GenericRecord> values, Context context) throws IOException, InterruptedException {
        for (GenericRecord record : values) {
            context.write(NullWritable.get(), record);
        }
    }
}
```

**优势**

- 分布式计算，高扩展性
- 容错性高

**劣势**

- 性能问题：处理该任务文件众多，磁盘I/O瓶颈，不适合大量的小文件处理

**时间估算**

- 假设有20个节点，没个节点并行处理100个任务
- 对于单个Map任务处理时间假设100ms，20个节点同时工作，集群每秒钟可以处理：20 * 100 =2000个，4M个任务处理时间 4M / 2000 = 33min
- 对于单个Reduce任务处理时间假设10ms，100个reducer，所以reduce时间：4M / 100 * 10ms = 6.67min
- 总时间约40分钟（理想时间不考虑集群调度，磁盘io，由于文件数量众多，MapReduce的磁盘io开销更大，所以时间会更长）

**优化方案**：使用spark集群



## 方案3: Spark分布式计算

**伪代码**

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("JSONL to Parquet") \
    .getOrCreate()

oss_path = "oss://bucket-name/path/to/jsonl/*"
parquet_path = "oss://bucket-name/path/to/output/parquet/"

df = spark.read.json(oss_path)
df.write.parquet(parquet_path)
```

**优势**

- 分布式计算，并行程度高
- 扩展性好，以通过增加节点数来扩展计算能力。

**劣势**

- 集群管理，维护复杂

**时间估算**

假设集群有100个4核 instance，单核处理一个文件需要100ms

文件处理时间 = 4M / 4 / 100 * 100ms = 16.67min （忽略集群调度、启动时间）



## 方案4: 阿里云函数计算

**处理代码：**

```python
import os
import pyarrow as pa
import pyarrow.parquet as pq
import oss2
import json
from io import BytesIO

def convert_jsonl_to_parquet(bucket, input_file_key, output_file_key):
    input_object = bucket.get_object(input_file_key)
    json_lines = input_object.read().decode('utf-8').splitlines()

    table_data = [json.loads(line) for line in json_lines]
    schema = pa.schema([pa.field(key, pa.string()) for key in table_data[0].keys()])
    table = pa.table(table_data, schema=schema)

    parquet_output = BytesIO()
    pq.write_table(table, parquet_output)
    parquet_output.seek(0)

    bucket.put_object(output_file_key, parquet_output)

def handler(event, context):
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, 'oss-cn-hangzhou.aliyuncs.com', bucket_name)

    result = bucket.list_objects(prefix="path/to/jsonl/")
    for obj in result.objects:
        input_file_key = obj.key
        output_file_key = input_file_key.replace("path/to/jsonl/", "path/to/parquet/").replace(".jsonl", ".parquet")

        convert_jsonl_to_parquet(bucket, input_file_key, output_file_key)
	
```

**优势**

- Serverless架构，自动扩展
- 开发简单，直接通过sdk访问资源

**劣势**

- 冷启动延迟，小文件I/O

**处理时间计算：**

**单个文件处理时间：**100 ms

**串行处理时间：** 4M * 100ms = 111h

**并行处理时间：**假设启动1000个实例并发，处理时间约为6分钟



# 2.并发推理数据

假设题目1里的 parquet 文件，里面 schema 为 prompt： string

现在需要你实现一个用类似 qwen coder 1.5b 作为 Reward Model 来给这些 prompt 打分。可以参考 https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 。

需要考虑：

1.如何能让这个模型推理速度最快？

- 比如对比 transformer 推理和 vLLM 推理的速度
- 海量数据要推理，怎么样能让单个推理实例的吞吐最大？

2.预估处理上述 4M 个 jsonl 文件，每个里 prompt 文本长度平均为 1k 情况下的处理速度



## 优化推理速度

1. 推理框架选择vLLM
2. 模型 4bit 量化
3. 动态批处理，最大化 GPU Mem 利用率
4. 多卡并行

```python
from vllm import LLM, SamplingParams
import json
import pyarrow.parquet as pq
import pandas as pd

def main():
    # 1. Path
    input_file = "input_file.parquet" 
    output_file = "output_file.jsonl"
    model_name = "Qwen/Qwen-1_5B"

    # 2. 加载 prompts
    prompts = load_prompts_from_parquet(input_file)

    # 3. 初始化 vLLM 模型
    llm = LLM(
        model=model_name,
        quantization="awq",     # 使用 4-bit AWQ 量化
        tensor_parallel_size=8, # 8 GPUs
        gpu_memory_utilization=0.98,
        max_model_len=1024,
    )

    # 4. 设置采样参数（仅生成一个标量分数）
    sampling_params = SamplingParams(
        temperature=0.0,  # 确定性输出
        max_tokens=1,     # 只生成一个得分token 
        stop=[],
    )

    # 5. 动态批处理推理
    outputs = llm.generate(prompts, sampling_params)

    # 6. 输出结果
    results = []
    for output in outputs:
        prompt = output.prompt
        score = output.outputs[0].text.strip()
        results.append({"prompt": prompt, "score": score})

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
```

**Q：怎么样能让单个推理实例的吞吐最大？**

1. 设置 `gpu_memory_utilization=0.9` 最大化显存利用率并防止显存溢出
2. 4-bit 量化 提高吞吐量
3. vLLM 中动态批处理和 PagedAttention 更高效利用显存



## 推理时间计算 

### 方法1 通过总 FLOPs 计算

**不考虑优化**

1. **Qwen 1.5B 模型的 FLOPs/token**：

   基于transformer的模型每个 token 单次前向传播的FLOPs可以近似表示为：**FLOPs ≈ 2 × 模型参数量（P）**
   $$
   \text{FLOPs/token} = 3 \times 10^9 \, \text{FLOPs/token}
   $$
   
2. **每个 prompt 的长度**：
   $$
   \text{每个 prompt 的长度} = 1k \, \text{tokens}
   $$
   
3. **每个 prompt 的计算量**：
   $$
   \text{每个 prompt 的计算量} = 3 \times 10^9 \times 1k = 3 \times 10^{12} \, \text{FLOPs}
   $$
   
4. **总任务的计算量**：
   $$
   \text{总任务的计算量} = 4M \times 3 \times 10^{12} = 1.2 \times 10^{19} \, \text{FLOPs}
   $$
   
5. **单卡 A100 80G 的吞吐量**：
   $$
   \text{单卡 A100 吞吐量} = 312 \, \text{TFLOPs/s} = 312 \times 10^{12} \, \text{FLOPs/s}
   $$
   
6. **8 卡 A100 的总吞吐量**：
   $$
   \text{8 卡 A100 总吞吐量} = 8 \times 312 \times 10^{12} = 2.496 \times 10^{15} \, \text{FLOPs/s}
   $$
   
7. **理论推理时间**

$$
\text{理论推理时间} = \frac{\text{总任务的计算量}}{\text{8 卡 A100 总吞吐量}}
$$

$$
\text{理论推理时间} = \frac{1.2 \times 10^{19}}{2.496 \times 10^{15}} \approx 4.81 \times 10^3 \, \text{秒} \approx 1.34 \, \text{小时}
$$



**考虑vLLM优化和int4量化**

我们假设：

- vLLM 加速带来的吞吐量提升为 2x
- INT4 量化能带来的吞吐量提升为 2x
- 假设 vLLM 加速和 INT4 量化的加速效果独立叠加

则综合的吞吐量提升可以通过将它们的加速效果相乘来计算：总加速比 = vLLM 加速比 × INT4 量化加速比  = 2 × 2           = 4

所以加速后的理论推理时间为1.34 / 4 =0.335 h = 20.1min



### 方法2 根据已有实验结果

根据qwen提供的[实验数据](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html)，在单卡 A100 80G GPU 环境下，相似模型`Qwen2.5-1.5B-Instruct` 推理速度如下表所示：

| 配置               | 量化方式 | 输入长度 (tokens) | 推理输出速率 (tokens/s) |
| ------------------ | -------- | ----------------- | ----------------------- |
| **无 vLLM 加速**   | BF16     | 6144              | 40.88                   |
| **采用 vLLM 加速** | AWQ      | 6144              | 203.64                  |

所以对于每个长度为 1k 的 prompt，每个 prompt 输出 1个 token（分数token）的情况下，使用8卡 A100 80G GPU 

- 使用原始 transformer 的推理时间估算为：
  $$
  \text{推理时间} = \frac{4,000,000}{40.88 \times 8} \approx 3.40 \, \text{小时}
  $$
  
- vLLM + AWQ 量化后 `qwen coder 1.5b` 的推理时间估算为：

$$
\text{推理时间} = \frac{4,000,000}{203.64 \times 8} \approx 40.9 \, \text{分钟}
$$