
目前 deepseek v2 moe 给出的实现中，只有 aux loss 和 lm loss。请根据论文中的公式，在现在 modeling_deepseek.py 的基础上实现其他两类 loss：Device-Level Balance Loss 和 Communication Balance Loss



改动说明：

修改https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py的MoEGate类
补充：
Device-Level Balance Loss 和 Communication Balance Loss 的计算
最终aux_loss为3者简单相加（代码：109-149行）

在config.json中添加了M
alpha 1, 2, 3 都使用aux_loss_alpha



核心代码：（代码：109-149行）

~~~
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
~~~