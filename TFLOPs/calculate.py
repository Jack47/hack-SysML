import json

def calculate_tflops(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    if "text_config" in config and "vision_config" in config:
        text_config = config["text_config"]
        vision_config = config["vision_config"]
    elif "text_cfg" in config and "vision_cfg" in config:
        text_config = config["text_cfg"]
        vision_config = config["vision_cfg"]

    hidden_size_text = text_config.get("hidden_size", 768) if "hidden_size" in text_config else text_config.get("width", 768)
    num_layers_text = text_config.get("num_hidden_layers", 12) if "num_hidden_layers" in text_config else text_config.get("num_layers", 12)
    # max_position_embeddings = text_config.get("max_position_embeddings", 77)
    max_position_embeddings = text_config.get("context_length", 77) if "context_length" in text_config else text_config.get("max_position_embeddings", 77)

    # 提取图像编码器配置
    hidden_size_image = vision_config.get("hidden_size", 1024) if "hidden_size" in vision_config else vision_config.get("width", 1024)
    num_layers_image = vision_config.get("num_hidden_layers", 24) if "num_hidden_layers" in vision_config else vision_config.get("layers", 24)
    image_size = vision_config.get("image_size", 336)
    patch_size = vision_config.get("patch_size", 14)
    mlp_ratio = vision_config.get("mlp_ratio", 4)

    # 计算序列长度
    seq_length_image = (image_size / patch_size) ** 2

    # 计算前馈网络中间层大小
    intermediate_size_text = hidden_size_text * mlp_ratio
    intermediate_size_image = hidden_size_image * mlp_ratio

    # 计算每层的运算量
    text_per_layer = (4 * hidden_size_text * max_position_embeddings ** 2 +
                      8 * hidden_size_text * intermediate_size_text * max_position_embeddings)
    image_per_layer = (4 * hidden_size_image * seq_length_image ** 2 +
                       8 * hidden_size_image * intermediate_size_image * seq_length_image)

    # 总运算量（乘以层数）
    total_ops = (text_per_layer * num_layers_text) + (image_per_layer * num_layers_image)

    # 转换为TFLOPs
    tflops = total_ops / 1e12
    return tflops

config_path = 'config_1.json'
tflops = calculate_tflops(config_path)
print(f"Calculated TFLOPs: {tflops}")

config_path = 'config_2.json'
tflops = calculate_tflops(config_path)
print(f"Calculated TFLOPs: {tflops}")

config_path = 'config_3.json'
tflops = calculate_tflops(config_path)
print(f"Calculated TFLOPs: {tflops}")
