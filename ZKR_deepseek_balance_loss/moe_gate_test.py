import math
import json


import torch
import torch.nn as nn 
import torch.nn.functional as F

"""
改动说明：

修改https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py的MoEGate类
补充：
Device-Level Balance Loss 和 Communication Balance Loss 的计算
最终aux_loss为3者简单相加（代码：109-149行）

在config.json中添加了M
alpha 1, 2, 3 都使用aux_loss_alpha

"""



class Config:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.data = json.load(file)

    def __getattr__(self, item):
        return self.data.get(item)
    


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        ) # [bs * seq_len, n_experts]
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)# [bs * seq_len, n_experts]
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "gready":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group] 标记出被激活的group
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) # [btz, seq_len*topk]
            if self.seq_aux:
                # Expert-Level Balance Loss.
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) # [btz, seq_len, e]
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                ) # [btz, 160]
                ce.scatter_add_( # 在被选中的60个expert上+1
                    1,
                    topk_idx_for_aux_loss, 
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device), # [2. 60]
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss_ExpertLevel = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha

                #  Device-Level Balance Loss.
                ce_ = ce.view(bsz, self.n_group, -1).mean(dim=-1) # [bsz,8]
                scores_for_seq_aux_ = scores_for_seq_aux.mean(dim=1).view(bsz, self.n_group, -1).mean(dim=-1)
                aux_loss_DeviceLevel = (ce_*scores_for_seq_aux_).sum(dim=1).mean() * self.alpha

                #  Communication Balance Loss. 
                # M = 256 
                M = self.config.M # ???? TODO 确认M是什么
                ce__ = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                ) # [btz, 160]
                ce__.scatter_add_( # 在被选中的60个expert上+1
                    1,
                    topk_idx_for_aux_loss, 
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device), # [2, 60]
                )
                ce__ = ce__.view(bsz, self.n_group, -1).sum(dim=-1).div_(M*seq_len/self.n_group) # [bsz, 8]
                aux_loss_Communication = (ce__*scores_for_seq_aux_).sum(dim=1).mean() * self.alpha

                aux_loss = aux_loss_ExpertLevel+aux_loss_DeviceLevel+aux_loss_Communication



            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss
    

cfg = Config('config.json')
X = torch.randn(2, 10, cfg.hidden_size)
moe_gate = MoEGate(cfg)
topk_idx, topk_weight, aux_loss = moe_gate(X)




