import sympy as sp
import json
import os
class FLOPsRecorder:
    def __init__(self,mul=0,add=0):
        """
        'mul' contains heavy float operator, such as multiplication/division/exponential function/square root/etc
        'add' contains light float operator, such as add/subtract
        """
        self.mul = mul
        self.add = add

    def __add__(self, other):
        if isinstance(other, FLOPsRecorder):
            mul = self.mul + other.mul
            add = self.add + other.add
            return FLOPsRecorder(mul,add)
        else:
            raise TypeError("Unsupported operand type.")

    def __mul__(self, other):
        if isinstance(other, sp.core.symbol.Symbol):
            mul = self.mul * other
            add = self.add * other
            return FLOPsRecorder(mul,add)
        else:
            raise TypeError("Unsupported operand type.")

    def __repr__(self):
        mul = sp.simplify(self.mul)
        add = sp.simplify(self.add)
        total = sp.simplify(self.total)

        mul_T = f"{mul / 1e12} TFLOPs" if len(mul.free_symbols)==0 else ""
        add_T = f"{add / 1e12} TFLOPs" if len(add.free_symbols)==0 else ""
        total_T = f"{total / 1e12} TFLOPs" if len(total.free_symbols)==0 else ""

        ls = [
            f"mul: {mul} FLOPs   {mul_T}",
            f"add: {add} FLOPs   {add_T}",
            f"total: {total} FLOPs   {total_T}"
        ]
        return "\n".join(ls)

    @property
    def total(self):
        return self.mul + self.add

    def subs(self, dic):
        mul = self.mul.subs(dic)
        add = self.add.subs(dic)
        return FLOPsRecorder(mul, add)

def get_bmm_flops(b,m,k,n):
    """
    b: batch size
    m,k: height and width of first matrix
    k,n: height and width of second matrix

    batched matrix multiplication:
    C[b,:,:] = A[b,:,:] * B[b,:,:]
    A \in R^{b,m,k}
    B \in R^{b,k,n}
    C \in R^{b,m,n}
    """
    return FLOPsRecorder(mul=b*m*n*k,add=b*m*n*(k-1))

def get_row_softmax_flops(b,m,n):
    """
    b: batch size
    m,n: height and width of the matrix.

    Do the softmax along row dimension
    """
    return FLOPsRecorder(mul=2*b*m*n,add=b*m*(n-1))

def get_attention_flops(b,s,d):
    flops_QKT = get_bmm_flops(b,s,d,s)
    flops_softmax = get_row_softmax_flops(b,s,s)
    flops_SV = get_bmm_flops(b,s,s,d)
    return flops_QKT + flops_softmax + flops_SV

def get_multi_head_attention_flops(b,s,d,h):
    """
    b: batch size
    s: sequence length
    d: hidden_size
    h: num_attention_heads
    """
    flops_per_head = get_attention_flops(b,s,d/h)
    flops = flops_per_head * h
    return flops

def get_projection_flops(b,s,di,do):
    """
    b: batch size
    s: sequence length
    di: input hidden_size
    do: output hidden_size

    projection: Y=WX+B
    X \in R^{b,s,di}
    W \in R^{do,di}
    Y \in R^{b,s,do}
    B \in R^{do}
    """
    return FLOPsRecorder(mul=b*s*do*di, add=b*s*do*(di-1)) + FLOPsRecorder(mul=0, add=b*s*do)

def get_multi_head_self_attention_flops(batch_size,seq_len,hidden_size,num_attention_heads):
    # Input Projection
    flops_Q = get_projection_flops(batch_size,seq_len,hidden_size,hidden_size)
    flops_scale = FLOPsRecorder(mul=batch_size*seq_len*hidden_size, add=0)
    flops_K = get_projection_flops(batch_size,seq_len,hidden_size,hidden_size)
    flops_V = get_projection_flops(batch_size,seq_len,hidden_size,hidden_size)
    flops_input_projection = flops_Q + flops_scale + flops_K + flops_V

    # MHA
    flops_hma = get_multi_head_attention_flops(batch_size,seq_len,hidden_size,num_attention_heads)

    # Output Projection
    flops_output_projection = get_projection_flops(batch_size,seq_len,hidden_size,hidden_size)

    return flops_input_projection + flops_hma + flops_output_projection

def get_sigmoid_flops(size):
    """
    y = 1/(1+exp(-x))
    x.shape is [size,]
    """
    return FLOPsRecorder(mul=2*size, add=size)

def get_gelu_flops(size):
    """
    y = 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    """
    pass

def get_quick_gelu_flops(size):
    """
    x * torch.sigmoid(1.702 * x)
    x.shape = [size,]
    """
    flops_pre_mul = FLOPsRecorder(mul=size, add=0)
    flops_sigmoid = get_sigmoid_flops(size)
    flops_post_mul = FLOPsRecorder(mul=size, add=0)
    return flops_pre_mul + flops_sigmoid + flops_post_mul

def get_mlp_flops(batch_size, seq_len, hidden_size, intermediate_size):
    flops_projection1 = get_projection_flops(batch_size, seq_len, hidden_size, intermediate_size)
    flops_quick_gelu = get_quick_gelu_flops(batch_size * seq_len * intermediate_size)
    flops_projection2 = get_projection_flops(batch_size, seq_len, intermediate_size, hidden_size)
    return flops_projection1 + flops_quick_gelu + flops_projection2

def get_mean_flops(batch_size, reduce_size):
    """
    Y[b] = mean(X[b,:])
    X.shape = [batch_size, reduce_size]
    Y.shape = [batch_size]

    mean:
    Y[b] = ( \sum_i X[b,i]) / reduce_size
    """
    return FLOPsRecorder(mul = batch_size, add = batch_size * (reduce_size - 1))

def get_var_flops(batch_size, reduce_size, has_mean = True):
    """
    Y[b] = var(X[b,:])
    X.shape = [batch_size, reduce_size]
    Y.shape = [batch_size]

    var(biased):
    Y[b] = ( \sum_i (X[b,i]-mean)^2 ) / reduce_size
    """
    if has_mean:
        flops_mean = FLOPsRecorder(mul=0,add=0)
    else:
        flops_mean = get_mean_flops(mul = batch_size, add = reduce_size)

    flops_diff = FLOPsRecorder(mul=0,add=batch_size * reduce_size)
    flops_square = FLOPsRecorder(mul=batch_size * reduce_size,add=0)
    flops_sum = FLOPsRecorder(mul=0,add=batch_size * (reduce_size-1))
    flops_div = FLOPsRecorder(mul=batch_size, add=0)
    return flops_mean + flops_diff + flops_square + flops_sum + flops_div

def get_elementwise_affine(size):
    return FLOPsRecorder(mul=size, add=size)

def get_layernorm_flops(batch_size, seq_len, hidden_size):
    """
    Y = (X-mean)/sqrt(var(X)+eps)
    Z = Y * weight + bias (element-wise affine transformation)

    Do layernorm along embedding dimension
    """
    flops_mean = get_mean_flops(batch_size * seq_len, hidden_size)
    flops_var = get_var_flops(batch_size * seq_len, hidden_size, has_mean=True)
    flops_eps = FLOPsRecorder(mul=0, add=batch_size * seq_len)
    flops_sqrt = FLOPsRecorder(mul=batch_size * seq_len, add=0)
    flops_standardize = FLOPsRecorder(mul=batch_size * seq_len * hidden_size, add=batch_size * seq_len * hidden_size)
    flops_elementwise_affine = get_elementwise_affine(batch_size * seq_len * hidden_size)
    return flops_mean + flops_var + flops_eps + flops_sqrt + flops_standardize + flops_elementwise_affine

def get_add_flops(size):
    return FLOPsRecorder(mul=0, add=size)

def get_encoderlayer_flops(batch_size, seq_len, hidden_size, intermediate_size, num_attention_heads):
    """
    identity = x
    x = layernorm(x)
    x = mhsa(x)
    x = identity + x
    identity = x
    x = layernorm(x)
    x = mlp(x)
    x = identity + x
    """
    flops_layernorm_1 = get_layernorm_flops(batch_size, seq_len, hidden_size)
    flops_mhsa = get_multi_head_self_attention_flops(batch_size, seq_len, hidden_size, num_attention_heads)
    flops_resadd_1 = get_add_flops(batch_size*seq_len*hidden_size)

    flops_layernorm_2 = get_layernorm_flops(batch_size, seq_len, hidden_size)
    flops_mlp = get_mlp_flops(batch_size, seq_len, hidden_size, intermediate_size)
    flops_resadd_2 = get_add_flops(batch_size*seq_len*hidden_size)

    return flops_layernorm_1 + flops_mhsa + flops_resadd_1 + flops_layernorm_2 + flops_mlp + flops_resadd_2

def get_encoder_flops(batch_size, seq_len, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers):
    """
    for i in range(num_hidden_layers):
        x = encoder_layer(x)
    """
    flops_encoder_layer = get_encoderlayer_flops(batch_size, seq_len, hidden_size, intermediate_size, num_attention_heads)
    return flops_encoder_layer * num_hidden_layers


def get_clip_text_embedding_flops(batch_size, seq_len, hidden_size):
    """
    The only float operation in CLIPTextEmbedding is 'add position_embedding'
    """
    return FLOPsRecorder(mul=0, add=batch_size*seq_len*hidden_size)
    
def get_clip_text_transformer_flops(batch_size, seq_len, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers):
    """
    x = text_embedding(input_ids),
    x = encoder(x)
    x = layernorm(x)
    """

    flops_clip_text_embedding = get_clip_text_embedding_flops(batch_size, seq_len, hidden_size)
    flops_encoder = get_encoder_flops(batch_size, seq_len, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers)
    flops_layernorm = get_layernorm_flops(batch_size, seq_len, hidden_size)
    return flops_clip_text_embedding + flops_encoder + flops_layernorm

def get_clip_vision_embedding_flops(batch_size, image_channel, image_size, patch_size, hidden_size):
    """
    
    """
    num_patches = (image_size/patch_size)**2
    flops_conv2d = get_bmm_flops(b = batch_size, m = num_patches, k = image_channel * patch_size * patch_size, n = hidden_size)
    seq_len = num_patches + 1
    flops_add_pos_embed = get_add_flops(size = batch_size * seq_len * hidden_size )
    return flops_conv2d + flops_add_pos_embed

def get_clip_vision_transformer_flops(batch_size, image_channel, image_size, patch_size, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers):
    """
    x = vision_embedding(x)
    x = layernorm(x)
    x = encoder(x)
    x = layernorm(x)
    """
    flops_clip_vision_embedding = get_clip_vision_embedding_flops(batch_size, image_channel, image_size, patch_size, hidden_size)

    num_patches = (image_size/patch_size)**2
    seq_len = num_patches + 1

    flops_pre_layernorm = get_layernorm_flops(batch_size, seq_len, hidden_size)
    flops_encoder = get_encoder_flops(batch_size, seq_len, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers)
    flops_post_layernorm = get_layernorm_flops(batch_size, seq_len, hidden_size)

    return flops_clip_vision_embedding + flops_pre_layernorm + flops_encoder + flops_post_layernorm

def get_l2norm_flops(batch_size, reduce_size):
    """
    Y[b] = \sqrt{\sum_i{X[b,i]^2}}
    Y.shape: batch_size
    X.shape: [batch_size, reduce_size]
    """
    flops_square = FLOPsRecorder(mul=batch_size * reduce_size, add=0)
    flops_sum = FLOPsRecorder(mul=0, add=batch_size * (reduce_size - 1))
    flops_sqrt = FLOPsRecorder(mul=batch_size, add=0)
    return flops_square + flops_sum + flops_sqrt


def get_matmul_flops(m, k, n):
    return FLOPsRecorder(mul=m*k*n, add=m*n*(k-1))

def get_clip_model_flops(projection_dim, text_symbol, vision_symbol):
    """
    x1 = text_model(x1)
    # x1.shape is [batch, hidden_size] now
    x1 = linear(x1, bias=False)  # text_projection
    x1 = x1 / x1.norm(p=2, dim=-1, keepdim=True) # normalize
    
    x2 = vision_model(x2)
    # x2.shape is [batch, hidden_size] now
    x2 = linear(x2, bias=False)  # vision_projection
    x2 = x2 / x2.norm(p=2, dim=-1, keepdim=True) # normalize

    logits_per_text = torch.matmul(x1, x2.t())
    logits_per_image = logits_per_text.t()
    """
    flops_text_model = get_clip_text_transformer_flops(batch_size=text_symbol["batch_size"], 
                                    seq_len=text_symbol["seq_len"], 
                                    hidden_size=text_symbol["hidden_size"], 
                                    intermediate_size=text_symbol["intermediate_size"], 
                                    num_attention_heads=text_symbol["num_attention_heads"], 
                                    num_hidden_layers=text_symbol["num_hidden_layers"])
    flops_text_projection = get_projection_flops(text_symbol["batch_size"],1,text_symbol["hidden_size"],projection_dim)
    flops_text_norm = get_l2norm_flops(text_symbol["batch_size"], projection_dim)
    flops_text_div = FLOPsRecorder(mul=text_symbol["batch_size"] * text_symbol["seq_len"] * projection_dim, add=0)

    flops_vision_model = get_clip_vision_transformer_flops(
                                    batch_size=vision_symbol["batch_size"], 
                                    image_channel=vision_symbol["image_channel"], 
                                    image_size=vision_symbol["image_size"], 
                                    patch_size=vision_symbol["patch_size"],
                                    hidden_size=vision_symbol["hidden_size"], 
                                    intermediate_size=vision_symbol["intermediate_size"], 
                                    num_attention_heads=vision_symbol["num_attention_heads"], 
                                    num_hidden_layers=vision_symbol["num_hidden_layers"])
    flops_vision_projection = get_projection_flops(vision_symbol["batch_size"],1,vision_symbol["hidden_size"],projection_dim)
    flops_vision_norm = get_l2norm_flops(vision_symbol["batch_size"], projection_dim)
    flops_vision_div = FLOPsRecorder(mul=vision_symbol["batch_size"] * projection_dim, add=0)

    flops_matmul = get_matmul_flops(m = text_symbol["batch_size"], k = projection_dim , n = vision_symbol["batch_size"] )

    return ( 
        flops_text_model + flops_text_projection + flops_text_norm + flops_text_div + 
        flops_vision_model + flops_vision_projection + flops_vision_norm + flops_vision_div + 
        flops_matmul
    )
    # return ( 
    #     flops_vision_model
    # )

def evaluation_with_model_arch(expr, config, projection_dim_symbol, text_symbol, vision_symbol):
    """
    expr: An expression composed of symbolic variables
    config: Model configuration file downloaded from huggingface
    projection_dim_symbol, text_symbol, vision_symbol: map name to symbolic variables
    """
    expr = expr.subs({
        projection_dim_symbol: config["projection_dim"],

        text_symbol["hidden_size"]: config["text_config_dict"]["hidden_size"],
        text_symbol["intermediate_size"]: config["text_config_dict"]["intermediate_size"],
        text_symbol["num_attention_heads"]: config["text_config_dict"]["num_attention_heads"],
        text_symbol["num_hidden_layers"]: config["text_config_dict"]["num_hidden_layers"],

        vision_symbol["hidden_size"]: config["vision_config"]["hidden_size"],
        vision_symbol["image_size"]: config["vision_config"]["image_size"],
        vision_symbol["intermediate_size"]: config["vision_config"]["intermediate_size"],
        vision_symbol["num_attention_heads"]: config["vision_config"]["num_attention_heads"],
        vision_symbol["num_hidden_layers"]: config["vision_config"]["num_hidden_layers"],
        vision_symbol["patch_size"]: config["vision_config"]["patch_size"],
        vision_symbol["image_channel"]: 3
    })
    return expr

def evaluation_with_model_arch_EVA(expr, config, projection_dim_symbol, text_symbol, vision_symbol):
    """
    EVA's config.json is different from 

    expr: An expression composed of symbolic variables
    config: Model configuration file from EVA
    projection_dim_symbol, text_symbol, vision_symbol: map name to symbolic variables
    """
    text_mlp_ratio = config["text_cfg"]["mlp_ratio"] if "mlp_ratio" in config["text_cfg"] else 4
    vision_mlp_ratio =  config["vision_cfg"]["mlp_ratio"] if "mlp_ratio" in config["vision_cfg"] else 4
    expr = expr.subs({
        projection_dim_symbol: config["embed_dim"],

        text_symbol["hidden_size"]: config["text_cfg"]["width"],
        text_symbol["intermediate_size"]: config["text_cfg"]["width"] * text_mlp_ratio,
        text_symbol["num_attention_heads"]: config["text_cfg"]["heads"],
        text_symbol["num_hidden_layers"]: config["text_cfg"]["layers"],

        vision_symbol["hidden_size"]: config["vision_cfg"]["width"],
        vision_symbol["image_size"]: config["vision_cfg"]["image_size"],
        vision_symbol["intermediate_size"]: config["vision_cfg"]["width"] * vision_mlp_ratio,
        vision_symbol["num_attention_heads"]: config["vision_cfg"]["width"] // config["vision_cfg"]["head_width"],
        vision_symbol["num_hidden_layers"]: config["vision_cfg"]["layers"],
        vision_symbol["patch_size"]: config["vision_cfg"]["patch_size"],
        vision_symbol["image_channel"]: 3
    })
    return expr

def evaluation_with_input(expr, input_text_config, input_vision_config, projection_dim_symbol, text_symbol, vision_symbol):
    expr = expr.subs({

        text_symbol["batch_size"]: input_text_config["batch_size"],
        text_symbol["seq_len"]: input_text_config["seq_len"],

        vision_symbol["batch_size"]: input_vision_config["batch_size"]
    })
    return expr

if __name__=="__main__":
    
    # create symbol variables
    projection_dim_symbol = sp.symbols("p")
    text_symbol = {
        "batch_size":sp.symbols("tb"),
        "seq_len":sp.symbols("ts"),
        "hidden_size":sp.symbols("td"),
        "intermediate_size":sp.symbols("tm"),
        "num_attention_heads":sp.symbols("th"),
        "num_hidden_layers":sp.symbols("tl")
    }
    vision_symbol = {
        "batch_size":sp.symbols("ib"),
        "image_channel":sp.symbols("ic"),
        "image_size":sp.symbols("iw"),
        "patch_size":sp.symbols("ip"),
        "hidden_size":sp.symbols("id"),
        "intermediate_size":sp.symbols("im"),
        "num_attention_heads":sp.symbols("ih"),
        "num_hidden_layers":sp.symbols("il")
    }
    flops_symbol = get_clip_model_flops(projection_dim_symbol, text_symbol, vision_symbol)
    print(flops_symbol)

    input_text_config = {
        "batch_size":128,
        "seq_len":64
    }
    input_vision_config = {
        "batch_size":128,
    }

    models = ["clip-vit-large-patch14-336", "CLIP-ViT-H-14-laion2B-s32B-b79K","EVA02_CLIP_E_psz14_s4B"]
    for model_name in models:
        with open(os.path.join("models",model_name,"config.json"), "r") as f:
            config = json.load(f)
        if "EVA" in model_name:
            flops_with_model_value = evaluation_with_model_arch_EVA(flops_symbol, config, projection_dim_symbol, text_symbol, vision_symbol)
        else:
            flops_with_model_value = evaluation_with_model_arch(flops_symbol, config, projection_dim_symbol, text_symbol, vision_symbol)
        # print(f"\n{model_name}'s FLOPs:")
        # print(flops_with_model_value)
        
        flops_with_input_value = evaluation_with_input(flops_with_model_value, input_text_config, input_vision_config, projection_dim_symbol, text_symbol, vision_symbol)
        print(f"\n{model_name}'s FLOPs under specific inputs:")
        print(flops_with_input_value)