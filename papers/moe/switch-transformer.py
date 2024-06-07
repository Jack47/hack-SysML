import mesh_tensorflow as mtf

# 只需要知道:
# 0. 每个 token 选择不同 expert 的权重: router_probs [num_cores, tokens_per_core, num_experts]
# 1. 每个 token 选择了(top1)哪些 expert: expert_mask [num_cores, tokens_per_core, num_experts] # 是 one-hot 形式

def load_balance_loss(router_probs, expert_mask):
    """Calculate load−balancing loss to ensure diverse expert routing."""
    # router probs is the probability assigned for each expert per token.
    # router probs shape: [num cores, tokens per core, num experts]
    # expert index contains the expert with the highest router probability in one−hot format.
    # expert mask shape: [num cores, tokens per core, num experts]
    # For each core, get the fraction of tokens routed to each expert.
    # density 1 shape: [num cores, num experts]
    # 每个 expert 上平均多少个(归一化之后的，求和是1)
    density_1 = mtf.reduce_mean(expert_mask, reduced_dim=tokens_per_core) # 即在 [, i, ] 上求均值
    # For each core, get fraction of probability mass assigned to each expert
    # from the router across all tokens.
    # density 1 proxy shape: [num cores, num experts]
    # 每个 expert 上平均的概率(归一化之后的，求和是1)
    density_1_proxy = mtf.reduce_mean(router_probs, reduced_dim=tokens_per_core)
    # density l for a single core: vector of length num experts that sums to 1.
    # density l proxy for a single core: vector of length num experts that sums to 1.
    # Want both vectors to have uniform allocation (1/num_experts) across all num expert elements.
    # The two vectors will be pushed towards uniform allocation when the dot product is minimized.
    # reduce_mean 没指定 reduced_dim，所以是所有元素的均值，即结果是标量
    loss = mtf.reduce_mean(density_1_proxy * density_1) * (num_experts^2)

    return loss

def router(inputs, expert_capacity):
    """Produce the combine and dispatch tensors used for sending and
    receiving tokens from their highest probability expert. """
    # Core layout is split across num cores for all tensors and operations.
    # inputs shape: [num cores, tokens per core, d model]
    
    router_weights = mtf.Variable(shape=[d model, num experts])
    # [num_cores, tokens_per_core, d_model] @ [d_model, num_experts] -> [num_cores, tokens_per_core, num_experts]
    # router logits shape: [num cores, tokens per core, num experts]
    router_logits = mtf.einsum([inputs, router_weights], reduced_dim=d model)
    if is_training:
        # Add noise for exploration across experts.
        router_logits += mtf.random uniform(shape=router logits.shape, minval=1−eps, maxval=1+eps)

    # Convert input to softmax operation from bfloat16 to float32 for stability.
    router_logits = mtf.to float32(router logits)
    # Probabilities for each token of what expert it should be sent to.
    router_probs = mtf.softmax(router logits, axis=−1)
    # Get the top−1 expert for each token. expert gate is the top−1 probability
    # from the router for each token. expert index is what expert each token
    # is going to be routed to.
    # expert gate shape: [num cores, tokens per core]
    # expert index shape: [num cores, tokens per core]
    expert gate, expert_index = mtf.top_1(router probs, reduced dim=num experts)
    # expert mask shape: [num cores, tokens per core, num experts]
    expert_mask = mtf.one_hot(expert_index, dimension=num_experts)
    # Compute load balancing loss.
    aux loss = load_balance_loss(router_probs, expert_mask)
    # Experts have a fixed capacity, ensure we do not exceed it. Construct
    # the batch indices, to each expert, with position in expert
    # make sure that not more that expert capacity examples can be routed to
    # each expert.
    # [, token_i, ] 上累加 -> [num_cores, tokens_per_core, num_experts]，即 [, tj , expert_i] 代表 expert_i 在 tj 时处理了多少 token
    # 也可以当作当前 token 在 expert_i 里的位置，如果超过 capacity 了，就会被处理掉

    position_in_expert = mtf.cumsum(expert_mask, dimension=tokens_per_core) ∗ expert_mask
    # Keep only tokens that fit within expert capacity.
    expert_mask ∗= mtf.less(position_in_expert, expert_capacity) # 干掉溢出的
    # [num_cores, tokens_per_core, num_experts] -> [num_cores, tokens_per_core] 因为本来是 one_hot，此时 sum 之后那就是 one_hot 的反向？
    expert_mask_flat = mtf.reduce_sum(expert mask, reduced_dim=experts_dim)
    # Mask out the experts that have overflowed the expert capacity.
    expert_gate ∗= expert_mask_flat
    # combine tensor used for combining expert outputs and scaling with router probability.
    # 用来收集专家输出的组合张量，并使用路由器概率进行缩放(加权）
    # 主要是 expert_gate(权重) * expert_mask(是否溢出) * expert_index(选了谁) * position_in_expert(处在被选的 expert 里 seq 里的哪个位置: seq index)
    # combine_tensor shape: [num cores, tokens per core, num_experts, expert_capacity]
    combine_tensor = (
    expert_gate ∗ expert mask flat ∗ # [num_cores, tokens_per_core] * [num_cores, tokens_per_core]
    mtf.one_hot(expert_index, dimension=num_experts) ∗ # -> [num_cores, tokens_per_core, num_experts]
    mtf.one_hot(position_in_expert, dimension=expert_capacity)) # [num_cores, tokens_per_core, num_experts] (one_hot)-> [num_cores, tokens_per_core, num_experts, expert_capacity]
    # Cast back outputs to bfloat16 for the rest of the layer.
    combine_tensor = mtf.to bfloat16(combine_tensor)
    # Create binary dispatch tensor that is 1 if the token gets routed to the corresponding expert.
    # dispatch tensor shape: [num cores, tokens per core, num experts, expert capacity]
    # dispatch 只关心是否要发给某个 expert，所以就不用知道是这个 expert 里的具体位置了。有点类似 one_hot
    dispatch_tensor = mtf.cast(combine tensor, tf.bool)
    return dispatch_tensor, combine_tensor, aux loss

def switch_layer(inputs, n, capacity_factor, num experts):
    """Distributed switch transformer feed−forward layer."""
    # num cores (n) = total cores for training the model (scalar).
    # d model = model hidden size (scalar).
    # num experts = total number of experts.
    # capacity factor = extra buffer for each expert.
    # inputs shape: [batch, seq len, d model]
    batch, seq len, d model = inputs.get shape()
    # Each core will route tokens per core tokens to the correct experts.
    tokens_per_core = batch ∗ seq len / num cores
    # Each expert will have shape [num cores, expert capacity, d model].
    # Each core is responsible for sending expert capacity tokens
    # to each expert.
    expert_capacity = tokens_per_core ∗ capacity_factor / num experts # 这里论文里就有问题，不应该再除以 num_experts 了，因为 capacity_factor 里已经除了 num_experts
    # Reshape to setup per core expert dispatching.
    # shape: [batch, seq len, d model] −> [num cores, tokens per core, d model]
    # Core layout: [n, 1, 1] −> [n, 1, 1]
    # 把 bsh 的 inputs 切分到 cores 上
    inputs = mtf.reshape(inputs, [num cores, tokens per core, d model])
    # Core Layout: [n, 1, 1] −> [n, 1, 1, 1], [n, 1, 1, 1]
    # dispatch tensor (boolean) shape: [num cores, tokens per core, num experts, expert capacity] # 是一个类似 one_hot 的
    # dispatch tensor is used for routing tokens to the correct expert.
    # combine tensor (float) shape: [num cores, tokens per core, num experts, expert capacity]
    # combine tensor used for combining expert outputs and scaling with router
    # probability.
    dispatch_tensor, combine_tensor, aux_loss = router(inputs, expert_capacity)
    # Matmul with large boolean tensor to assign tokens to the correct expert.
    # Core Layout: [n, 1, 1], −> [1, n, 1, 1]
    # [num cores, tokens per core, d_model] @ [num cores, tokens_per_core, num experts, expert capacity]
    # -> [num cores, tokens per core, num experts, expert capacity]
    # expert inputs shape: [num experts, num_cores, expert_capacity, d model]
    # 没搞明白这个步骤维度怎么变化的
    expert_inputs = mtf.einsum([inputs, dispatch_tensor], reduce_dims=[tokens_per_core])
    # 这里省略了 all2all，所以下面的 mtf.reshape 是直接模拟的 all2all 之后的结果？
    # All−to−All communication. Cores split across num cores and now we want to split
    # across num experts. This sends tokens, routed locally, to the correct expert now
    # split across different cores.
    # Core layout: [1, n, 1, 1] −> [n, 1, 1, 1]
    expert inputs = mtf.reshape(expert inputs, [num experts, num cores, expert capacity, d model])
    # expert_capacity 这个维度上，FFN 计算时并没有考虑 batch 和 顺序，因为只需要计算出来，之后 all2all 返回完，会再重新拼起来
    # 所以 FFN 前是固定某种顺序就行。比如 batch 已经混合在 b*s 里了。而 all2all 时到达的顺序就是根据 rank 来的？
    # Standard feed forward computation, where each expert will have its own
    # unique set of parameters.
    # Total unique parameters created: num experts ∗ (d model ∗ d ff ∗ 2).
    # expert outputs shape: [num experts, num cores, expert capacity, d model]
    expert outputs = feed forward(expert inputs)
    # All−to−All communication. Cores are currently split across the experts
    # dimension, which needs to be switched back to being split across num cores.
    # Core Layout: [n, 1, 1, 1] −> [1, n, 1, 1]
    expert outputs = mtf.reshape(expert outputs, [num experts, num cores, expert capacity, d model])
    # Convert back to input shape and multiply outputs of experts by the routing probability.
    # expert outputs shape: [num experts, num cores, tokens per core, d model]
    # expert outputs combined shape: [num cores, tokens per core, d model]
    # Core Layout: [1, n, 1, 1] −> [n, 1, 1]
    expert outputs combined = mtf.einsum([expert outputs, combine tensor], reduce dims=[tokens per core])
    # Remove tokens per core shapes used for local routing dispatching to match input shape.
    # Core Layout: [n, 1, 1] −> [n, 1, 1]
    outputs = mtf.reshape(expert outputs combined, [batch, seq len, d model])
    return outputs, aux loss