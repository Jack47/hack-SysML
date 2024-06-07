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
    density_1 = mtf.reduce_mean(expert_mask, reduced_dim=tokens_per_core)
    # For each core, get fraction of probability mass assigned to each expert
    # from the router across all tokens.
    # density 1 proxy shape: [num cores, num experts]
    # 每个 expert 上平均的概率(归一化之后的，求和是1)
    density_1_proxy = mtf.reduce_mean(router_probs, reduced_dim=tokens_per_core)
    # density l for a single core: vector of length num experts that sums to 1.
    # density l proxy for a single core: vector of length num experts that sums to 1.
    # Want both vectors to have uniform allocation (1/num_experts) across all num expert elements.
    # The two vectors will be pushed towards uniform allocation when the dot product is minimized.
    # dot product 的结果是标量
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
    # [num_cores, tokens_per_core, num_experts]
    # 问题：这里维度只需要 2 维? [num_cores, num_experts]，不需要维度的这个序列
    position_in_expert = mtf.cumsum(expert_mask, dimension=tokens_per_core) ∗ expert_mask
    # Keep only tokens that fit within expert capacity.
    expert_mask ∗= mtf.less(position_in_expert, expert_capacity) # 干掉溢出的
    expert_mask_flat = mtf.reduce_sum(expert mask, reduced_dim=experts_dim) # [num_cores, tokens_per_core, num_experts] -> [num_cores, tokens_per_core]
    # Mask out the experts that have overflowed the expert capacity.
    expert gate ∗= expert mask flat
    # combine tensor used for combining expert outputs and scaling with router probability.
    # 用来收集专家输出的组合张量，并使用路由器概率进行缩放(加权）
    # 主要是 expert_gate(权重) * expert_mask(是否溢出) * expert_index(选了谁) * position_in_expert(在哪个位置: seq index)
    # combine_tensor shape: [num cores, tokens per core, num_experts, expert_capacity]
    combine_tensor = (
    expert gate ∗ expert mask flat ∗ # [num_cores, tokens_per_core] * [num_cores, tokens_per_core]
    mtf.one_hot(expert_index, dimension=num_experts) ∗ # -> [num_cores, tokens_per_core, num_experts]
    mtf.one_hot(position_in_expert, dimension=expert_capacity)) # [num_cores, tokens_per_core, num_experts] (one_hot)-> [num_cores, tokens_per_core, num_experts, expert_capacity]
    # Cast back outputs to bfloat16 for the rest of the layer.
    combine_tensor = mtf.to bfloat16(combine tensor)
    # Create binary dispatch tensor that is 1 if the token gets routed to the corresponding expert.
    # dispatch tensor shape: [num cores, tokens per core, num experts, expert capacity]
    # dispatch 只关心是否要发给某个 expert，所以就不用知道是这个 expert 里的具体位置了
    dispatch_tensor = mtf.cast(combine tensor, tf.bool)
    return dispatch tensor, combine tensor, aux loss