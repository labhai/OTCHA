# loss/otra_loss.py
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualDistillationLoss(nn.Module):
    def __init__(self, temp=4., lambda_hyperparam=.1):
        super(MutualDistillationLoss, self).__init__()
        self.temp = temp
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.lambda_hyperparam = lambda_hyperparam

    def forward(self, multi_view_logits, single_view_logits, targets):
        averaged_single_logits = torch.mean(single_view_logits, dim=1)
        
        q = torch.softmax(averaged_single_logits / self.temp, dim=1)
        try:
            max_q, pred_q = torch.max(q, dim=1)
            q_correct = pred_q == targets
            q_correct = q_correct.float().mean().item()
            max_q = max_q.mean().item()
        except RuntimeError:
            q_correct = 0.
            max_q = 0.
        p = torch.softmax(multi_view_logits / self.temp, dim=1)
        max_p, _ = torch.max(p, dim=1)
        max_p = max_p.mean().item()

        log_q = torch.log_softmax(averaged_single_logits / self.temp, dim=1)
        log_p = torch.log_softmax(multi_view_logits / self.temp, dim=1)
        
        loss = (1/2) * (self.kl_div(log_p, q.detach()).sum(dim=1).mean() + self.kl_div(log_q, p.detach()).sum(dim=1).mean())
        loss_weighted = loss * (self.temp ** 2) * self.lambda_hyperparam

        return loss_weighted

    
class OTRALoss(nn.Module):
    """Confidence-weighted cosine alignment with stop-gradient hub messages."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, output: Dict[str, Any]) -> torch.Tensor:
        info = output["otcha"]
        student = info["tokens_refined"]
        teacher = info["messages"].detach()
        weights = (info["gate"] * info["beta"].unsqueeze(-1)).detach().float()

        # Normalize in float32 for stable cosine distances under autocast.
        student = F.normalize(F.layer_norm(student.float(), (student.shape[-1],), eps=1e-6), dim=-1)
        teacher = F.normalize(F.layer_norm(teacher.float(), (teacher.shape[-1],), eps=1e-6), dim=-1)
        distance = 1.0 - (student * teacher).sum(dim=-1)
        return (weights * distance).sum() / (weights.sum() + self.eps)
