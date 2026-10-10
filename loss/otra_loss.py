# loss/otra_loss.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Sequence, Union

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
    def __init__(
        self,
        student_key: str = "tokens_refined",
        teacher_key: str = "messages",
        weight_mode: str = "gate_beta",  # gate_beta, gate_beta_confv, confv_beta, gate, beta, uniform
        detach_teacher: bool = True,
        detach_weights: bool = True,
        feature_norm: str = "layernorm+l2",  # layernorm+l2, l2, none
        distance: str = "cosine",  # cosine, mse, smooth_l1
        eps: float = 1e-8,
        debug_checks: bool = False,
        # ---- stage-aware options ----
        use_otcha_stages: bool = True,
        stages_key: str = "otcha_stages",
        stage_reduce: str = "mean",          # mean | sum
        stage_start: int = 0,                # 0: include stage0, 1: mid-stages only
        stage_end: Optional[int] = None,     # None: all
        stage_weights: Optional[Sequence[float]] = None,  # explicit weights (after slicing)
        stage_weight_decay: float = 1.0,     # if stage_weights is None: w_i = decay^i
    ):
        super().__init__()
        self.student_key = student_key
        self.teacher_key = teacher_key
        self.weight_mode = weight_mode
        self.detach_teacher = detach_teacher
        self.detach_weights = detach_weights
        self.feature_norm = feature_norm
        self.distance = distance
        self.eps = float(eps)
        self.debug_checks = bool(debug_checks)

        self.use_otcha_stages = bool(use_otcha_stages)
        self.stages_key = str(stages_key)
        self.stage_reduce = str(stage_reduce)
        self.stage_start = int(stage_start)
        self.stage_end = int(stage_end) if stage_end is not None else None
        self.stage_weights = list(stage_weights) if stage_weights is not None else None
        self.stage_weight_decay = float(stage_weight_decay)

        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)

        valid_weight_modes = {"gate_beta", "gate_beta_confv", "confv_beta", "gate", "beta", "uniform"}
        if self.weight_mode not in valid_weight_modes:
            raise ValueError(f"Invalid weight_mode: {self.weight_mode}. Choose from {sorted(list(valid_weight_modes))}")

        valid_norms = {"layernorm+l2", "l2", "none"}
        if self.feature_norm not in valid_norms:
            raise ValueError(f"Invalid feature_norm: {self.feature_norm}. Choose from {sorted(list(valid_norms))}")

        valid_dist = {"cosine", "mse", "smooth_l1"}
        if self.distance not in valid_dist:
            raise ValueError(f"Invalid distance: {self.distance}. Choose from {sorted(list(valid_dist))}")

        if self.stage_reduce not in {"mean", "sum"}:
            raise ValueError("stage_reduce must be one of {'mean','sum'}")

        if self.stage_weight_decay <= 0:
            raise ValueError("stage_weight_decay must be > 0")

        if self.stage_start < 0:
            raise ValueError("stage_start must be >= 0")

    def forward(self, obj: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]) -> torch.Tensor:
        stages = self._extract_stages(obj)
        if len(stages) == 0:
            return self._zero

        # slice
        if self.stage_end is None:
            stages = stages[self.stage_start :]
        else:
            stages = stages[self.stage_start : self.stage_end]

        if len(stages) == 0:
            return self._zero

        stage_ws = self._get_stage_weights(len(stages))

        losses = []
        wsum = 0.0
        for si, w in zip(stages, stage_ws):
            li = self._forward_one_stage(si)
            if li is None:
                continue
            losses.append(li * float(w))
            wsum += float(w)

        if len(losses) == 0:
            return self._zero

        total = torch.stack(losses, dim=0).sum()
        if self.stage_reduce == "mean":
            total = total / (wsum + self.eps)

        return total

    def _extract_stages(self, obj: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        if obj is None:
            return []

        if isinstance(obj, (list, tuple)):
            return [s for s in obj if isinstance(s, dict)]

        if not isinstance(obj, dict):
            return []

        if self.use_otcha_stages and (self.stages_key in obj) and isinstance(obj[self.stages_key], (list, tuple)):
            stages = [s for s in obj[self.stages_key] if isinstance(s, dict)]
            if len(stages) > 0:
                return stages

        if "otcha" in obj and isinstance(obj["otcha"], dict):
            return [obj["otcha"]]

        if "edges" in obj:
            return [obj]

        return []

    def _get_stage_weights(self, num_stages: int) -> List[float]:
        if self.stage_weights is not None:
            if len(self.stage_weights) != num_stages:
                raise ValueError(f"stage_weights length mismatch: got {len(self.stage_weights)} but need {num_stages}.")
            return [float(w) for w in self.stage_weights]

        return [self.stage_weight_decay ** i for i in range(num_stages)]

    def _forward_one_stage(self, otcha: Dict[str, Any]) -> Optional[torch.Tensor]:
        if otcha is None:
            return None
        if self.student_key not in otcha or self.teacher_key not in otcha or "edges" not in otcha:
            return None

        tokens = otcha[self.student_key]   # (B,N,S,D)
        teacher = otcha[self.teacher_key]  # (B,E,S,D)
        edges = otcha["edges"]             # (E,2)

        if self.debug_checks:
            if not isinstance(tokens, torch.Tensor) or tokens.dim() != 4:
                raise ValueError(f"{self.student_key} must be (B,N,S,D), got {type(tokens)} {getattr(tokens,'shape',None)}")
            if not isinstance(teacher, torch.Tensor) or teacher.dim() != 4:
                raise ValueError(f"{self.teacher_key} must be (B,E,S,D), got {type(teacher)} {getattr(teacher,'shape',None)}")
            if not isinstance(edges, torch.Tensor) or edges.dim() != 2 or edges.shape[1] != 2:
                raise ValueError(f"edges must be (E,2), got {type(edges)} {getattr(edges,'shape',None)}")

        B, N, S, D = tokens.shape
        _, E, S2, D2 = teacher.shape

        if self.debug_checks:
            if S2 != S or D2 != D:
                raise ValueError(f"Teacher shape mismatch: tokens S,D=({S},{D}) vs teacher ({S2},{D2})")
            if edges.shape[0] != E:
                raise ValueError(f"edges E mismatch: edges has {edges.shape[0]} but teacher has {E}")

        v_idx = edges[:, 1]  # (E,)
        student = tokens[:, v_idx]          # (B,E,S,D)

        if self.detach_teacher:
            teacher = teacher.detach()

        w = self._build_weights(otcha, B=B, E=E, S=S, device=student.device, dtype=student.dtype)
        if self.detach_weights:
            w = w.detach()

        w_sum = w.sum()
        # NOTE:
        if (not torch.isfinite(w_sum).item()) or (w_sum.item() < self.eps):
            return None

        student_f = self._normalize_features(student.float())
        teacher_f = self._normalize_features(teacher.float())
        w_f = w.float()

        per = self._per_token_distance(student_f, teacher_f)  # (B,E,S)
        loss = (w_f * per).sum() / (w_f.sum() + self.eps)
        return loss

    def _build_weights(self, otcha: Dict[str, Any], B: int, E: int, S: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.weight_mode == "uniform":
            return torch.ones((B, E, S), device=device, dtype=dtype)

        gate = otcha.get("gate", None)      # (B,E,S)
        beta = otcha.get("beta", None)      # (B,E)
        conf_v = otcha.get("conf_v", None)  # (B,E,S)

        if self.weight_mode == "gate_beta":
            if gate is None or beta is None:
                return torch.ones((B, E, S), device=device, dtype=dtype)
            return gate * beta.unsqueeze(-1)

        if self.weight_mode == "gate_beta_confv":
            if gate is None or beta is None or conf_v is None:
                if gate is not None and beta is not None:
                    return gate * beta.unsqueeze(-1)
                return torch.ones((B, E, S), device=device, dtype=dtype)
            return gate * beta.unsqueeze(-1) * conf_v

        if self.weight_mode == "confv_beta":
            if conf_v is None or beta is None:
                return torch.ones((B, E, S), device=device, dtype=dtype)
            return conf_v * beta.unsqueeze(-1)

        if self.weight_mode == "gate":
            if gate is None:
                return torch.ones((B, E, S), device=device, dtype=dtype)
            return gate

        if self.weight_mode == "beta":
            if beta is None:
                return torch.ones((B, E, S), device=device, dtype=dtype)
            return beta.unsqueeze(-1).expand(B, E, S)

        return torch.ones((B, E, S), device=device, dtype=dtype)

    def _normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_norm == "none":
            return x
        if self.feature_norm == "l2":
            return F.normalize(x, dim=-1)
        if self.feature_norm == "layernorm+l2":
            D = x.shape[-1]
            x = F.layer_norm(x, (D,), weight=None, bias=None, eps=1e-6)
            x = F.normalize(x, dim=-1)
            return x
        return x

    def _per_token_distance(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if self.distance == "cosine":
            sim = (student * teacher).sum(dim=-1)  # (B,E,S)
            return 1.0 - sim
        if self.distance == "mse":
            return (student - teacher).pow(2).sum(dim=-1)
        if self.distance == "smooth_l1":
            return F.smooth_l1_loss(student, teacher, reduction="none").sum(dim=-1)
        return (student - teacher).pow(2).sum(dim=-1)
