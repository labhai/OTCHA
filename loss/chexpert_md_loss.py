from __future__ import annotations

from typing import Any, Dict, Optional, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualDistillationLoss(nn.Module):
    def __init__(self, temp: float = 4.0, lambda_hyperparam: float = 0.1):
        super().__init__()
        self.temp = temp
        self.lambda_hyperparam = lambda_hyperparam

    @staticmethod
    def _binary_kl(logits_p: torch.Tensor, logits_q: torch.Tensor, temp: float) -> torch.Tensor:
        lp = logits_p / temp
        lq = logits_q / temp

        log_p = F.logsigmoid(lp)
        log_1mp = F.logsigmoid(-lp)
        log_q = F.logsigmoid(lq)
        log_1mq = F.logsigmoid(-lq)
        q = torch.sigmoid(lq)

        kl = q * (log_q - log_p) + (1.0 - q) * (log_1mq - log_1mp)
        return kl

    def forward(self, multi_view_logits: torch.Tensor,
                single_view_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        avg_single = single_view_logits.mean(dim=1)  # [B, 13]

        # KL(q=single || p=multi_view): single detached, multi_view learns
        kl_fwd = self._binary_kl(multi_view_logits, avg_single.detach(), self.temp)
        # KL(q=multi_view || p=single): multi_view detached, single learns
        kl_rev = self._binary_kl(avg_single, multi_view_logits.detach(), self.temp)

        loss = 0.5 * (kl_fwd.sum(dim=1).mean() + kl_rev.sum(dim=1).mean())
        return loss * (self.temp ** 2) * self.lambda_hyperparam