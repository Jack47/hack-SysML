import mesh_tensorflow as mtf
def router(inputs, capacity_factor):
    """Produce the combine and dispatch tensors used for sending and
    receiving tokens from their highest probability expert. """
    # Core layout is split across num cores for all tensors and operations.
    # inputs shape: [num cores, tokens per core, d model]
    
    router_weights = mtf.Variable(shape=[d model, num experts])
    # router logits shape: [num cores, tokens per core, num experts]
    router_logits = mtf.einsum([inputs, router_weights], reduced_dim=d model)
    if is_training:
        # Add noise for exploration across experts.
        router_logits += mtf.random uniform(shape=router logits.shape, minval=1−eps, maxval=1+eps)

    # Convert input to softmax operation from bfloat16 to float32 for stability.
    router_logits = mtf.to float32(router logits)
    # Probabilities for each token of what expert it should be sent to.
    router probs = mtf.softmax(router logits, axis=−1)
    # Get the top−1 expert for each token. expert gate is the top−1 probability
    # from the router for each token. expert index is what expert each token
    # is going to be routed to.
    # expert gate shape: [num cores, tokens per core]
    # expert index shape: [num cores, tokens per core]
    expert gate, expert index = mtf.top 1(router probs, reduced dim=num experts)
    # expert mask shape: [num cores, tokens per core, num experts]
    expert mask = mtf.one hot(expert index, dimension=num experts)
    # Compute load balancing loss.
    aux loss = load balance loss(router probs, expert mask)
    # Experts have a fixed capacity, ensure we do not exceed it. Construct
    # the batch indices, to each expert, with position in expert
    # make sure that not more that expert capacity examples can be routed to
    # each expert.
    position in expert = mtf.cumsum(expert mask, dimension=tokens per core) ∗ expert mask
    # Keep only tokens that fit within expert capacity.
    expert mask ∗= mtf.less(position in expert, expert capacity)
    expert mask flat = mtf.reduce sum(expert mask, reduced dim=experts dim)
    # Mask out the experts that have overflowed the expert capacity.
    expert gate ∗= expert mask flat
    # combine tensor used for combining expert outputs and scaling with router probability.
    # combine tensor shape: [num cores, tokens per core, num experts, expert capacity]
    combine tensor = (
    expert gate ∗ expert mask flat ∗
    mtf.one hot(expert index, dimension=num experts) ∗
    mtf.one hot(position in expert, dimension=expert capacity))
    # Cast back outputs to bfloat16 for the rest of the layer.
    combine tensor = mtf.to bfloat16(combine tensor)
    # Create binary dispatch tensor that is 1 if the token gets routed to the corresponding expert.
    # dispatch tensor shape: [num cores, tokens per core, num experts, expert capacity]
    dispatch tensor = mtf.cast(combine tensor, tf.bool)
    return dispatch tensor, combine tensor, aux loss