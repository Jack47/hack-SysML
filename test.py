import torch
n_routed_experts = 160
n_group = 8
device = torch.device("cuda:0")
bsz = 4
seq_len = 64
topk = 6
topk_group = 3  # 最大
alpha1, alpha2, alpha3 = 0.001, 0.001, 0.001
def test(scores_for_seq_aux, topk_idx_for_aux_loss):
    ce = torch.zeros(
        bsz, n_routed_experts, device=device
    )
    ce.scatter_add_(
        1,
        topk_idx_for_aux_loss,
        torch.ones(bsz, seq_len * topk, device=device),
    ).div_(seq_len * topk / n_routed_experts)  # ce: Tensor[bsz, num_experts]
    aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
        dim=1
    ).mean() * alpha1

    # TODO: 实现device_loss，假设按顺序将专家分组
    # [bsz, num_experts] -> [bsz, n_groups]
    experts_per_device = n_routed_experts // n_group
    ce_groups = ce.view(bsz, n_group, experts_per_device).mean(dim=-1)
    # [bsz, seq_len, num_experts] -> [bsz, num_experts] -> [bsz, n_groups]
    p_groups = scores_for_seq_aux.mean(dim=1).view(bsz, n_group, experts_per_device).sum(dim=-1)
    # [bsz, num_groups] -> [bsz] -> value
    device_loss = (ce_groups * p_groups).sum(dim=1).mean() * alpha2

    # TODO: 实现comm_loss
    ce_comm = torch.zeros(
        bsz, n_routed_experts, device=device
    )
    # [bsz, n_groups]
    ce_comm = ce_comm.scatter_add_(
        1,
        topk_idx_for_aux_loss,
        torch.ones(bsz, seq_len * topk, device=device),
    ).view(bsz, n_group, experts_per_device).sum(dim=-1).div_(seq_len * topk_group / n_group)
    p_comm = p_groups
    comm_loss = (ce_comm * p_comm).sum(dim=1).mean() * alpha3
    return aux_loss, device_loss, comm_loss

if __name__ == '__main__':
    # [bsz, seq_len, n_routed_experts]
    scores_for_seq_aux = torch.randn((bsz, seq_len, n_routed_experts), device=device).softmax(dim=-1)
    # [bsz, topk * seq_len]，皆为[0, n_routed_experts-1]的整数索引
    topk_weight, topk_idx = torch.topk(
        scores_for_seq_aux, k=topk, dim=-1, sorted=False
    )
    topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
    test_aux_loss, test_device_loss, test_comm_loss = test(scores_for_seq_aux=scores_for_seq_aux, topk_idx_for_aux_loss=topk_idx_for_aux_loss)
    print(f'expert_loss: {test_aux_loss}, device_loss: {test_device_loss}, comm_loss: {test_comm_loss}')
