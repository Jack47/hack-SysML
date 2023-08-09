参考的这个：https://docs.openvino.ai/2023.0/notebooks/107-speech-recognition-quantization-data2vec-with-output.html#check-int8-model-inference-result


执行引擎是 openvino，而 nncf 是 intel 的 neuron network compression framework for OpenVINO inference.

把 pytorch 模型转换为 openvino 的格式，而且是 fp16的(model.xml和model.bin）

pytorch->onnx->xml(fp16)->nncf quantize->xml(int8)

流程如下图：![](imgs/openvino-basic-flow_simplified.svg)

```
from openvino.tools import mo
from openvino.runtime import serialize, Core
import torch

ov_model = mo.convert_model(onnx_model_path, compress_to_fp16=True) # 转模型时修改为 fp16
serialize(ov_model, str(ir_model_path)) # 此时就序列化为 openvino 的 IR？
ov_ir_model = core.read_model(ir_model_path)

core = Core()
compiled_model = core.compile_model(ov_model)
ov_transcription = ov_infer(compiled_model, dataset[0])

# 还可以去对比精度：使用 pytorch vs openvino

import nncf
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.range_estimator import AggregatorType
from nncf.quantization.range_estimator import StatisticsType

from nncf.parameters import ModelType

# 怎么选择不同的量化方式呢？比如 int8，int4
quantized_model = nncf.quantize(
    ov_model,
    calibration_dataset,
    model_type = ModelType.TRANSFORMER,
    subset_size=len(dataset),
    ignored_scope = nncf.IgnoredScope(
    ),
    advanced_parameters=AdvancedQuantizationParameters()
## inference int8 model
int8_compiled_model = core.compile_model(quantized_model) # 把 IR model 转换为可以直接推理的 openvino 模型
```

自带 benchmark 工具，可以衡量 FP16 和 INT8 模型的精度

