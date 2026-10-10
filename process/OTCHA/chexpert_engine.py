from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import einops

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        precision_score, recall_score, average_precision_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Default CheXpert labels ─────────────────────────────────────────────────
CHEXPERT_LABELS: List[str] = [
    "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]


# =============================================================================
# MetricsResult
# =============================================================================
@dataclass
class MetricsResult:
    mean_auc: float = 0.0
    mean_ap: float = 0.0
    mean_acc: float = 0.0
    mean_f1: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    loss: float = 0.0
    num_samples: int = 0
    per_label: Dict[str, Dict[str, float]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "mean_auc": self.mean_auc,
            "mean_ap": self.mean_ap,
            "mean_acc": self.mean_acc,
            "mean_f1": self.mean_f1,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "loss": self.loss,
            "num_samples": self.num_samples,
        }
        for label_name, metrics in self.per_label.items():
            for metric_name, val in metrics.items():
                d[f"{label_name}/{metric_name}"] = val
        d.update({f"extra/{k}": v for k, v in self.extra.items()})
        return d

    def __str__(self) -> str:
        return (
            f"mAUC: {self.mean_auc:.4f} | "
            f"mAP: {self.mean_ap:.4f} | "
            f"mAcc: {self.mean_acc:.4f} | "
            f"mF1: {self.mean_f1:.4f} | "
            f"Loss: {self.loss:.4f}"
        )


# =============================================================================
# MultilabelMetricsComputer
# =============================================================================
class MultilabelMetricsComputer:
    def __init__(self, label_names: Optional[List[str]] = None, threshold: float = 0.5):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required.")
        self.label_names = label_names or CHEXPERT_LABELS
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.all_probs: List[np.ndarray] = []
        self.all_preds: List[np.ndarray] = []
        self.all_target: List[np.ndarray] = []
        self.loss_sum: float = 0.0
        self.num_samples: int = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor, loss: float = 0.0) -> None:
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= self.threshold).astype(int)
        target_np = target.detach().cpu().numpy()

        self.all_probs.append(probs)
        self.all_preds.append(preds)
        self.all_target.append(target_np)
        self.loss_sum += loss * target_np.shape[0]
        self.num_samples += target_np.shape[0]

    def compute(self) -> MetricsResult:
        if self.num_samples == 0:
            return MetricsResult()

        probs = np.concatenate(self.all_probs)    # [N, K]
        preds = np.concatenate(self.all_preds)    # [N, K]
        target = np.concatenate(self.all_target)  # [N, K]
        num_labels = target.shape[1]

        per_label_auc = []
        per_label_ap = []
        per_label_acc = []
        per_label_f1 = []
        per_label_prec = []
        per_label_rec = []

        for i in range(num_labels):
            t_i, p_i, pr_i = target[:, i], preds[:, i], probs[:, i]

            # AUC — 한 class만 존재하면 계산 불가
            try:
                per_label_auc.append(roc_auc_score(t_i, pr_i))
            except ValueError:
                per_label_auc.append(0.0)

            # AP
            try:
                per_label_ap.append(average_precision_score(t_i, pr_i))
            except ValueError:
                per_label_ap.append(0.0)

            per_label_acc.append(accuracy_score(t_i, p_i))
            per_label_f1.append(f1_score(t_i, p_i, zero_division=0))
            per_label_prec.append(precision_score(t_i, p_i, zero_division=0))
            per_label_rec.append(recall_score(t_i, p_i, zero_division=0))

        # per-label dict
        per_label = {}
        for i, name in enumerate(self.label_names[:num_labels]):
            per_label[name] = {
                "auc": per_label_auc[i],
                "ap": per_label_ap[i],
                "acc": per_label_acc[i],
                "f1": per_label_f1[i],
                "precision": per_label_prec[i],
                "recall": per_label_rec[i],
            }

        return MetricsResult(
            mean_auc=float(np.mean(per_label_auc)),
            mean_ap=float(np.mean(per_label_ap)),
            mean_acc=float(np.mean(per_label_acc)),
            mean_f1=float(np.mean(per_label_f1)),
            mean_precision=float(np.mean(per_label_prec)),
            mean_recall=float(np.mean(per_label_rec)),
            loss=self.loss_sum / self.num_samples,
            num_samples=self.num_samples,
            per_label=per_label,
        )


# =============================================================================
# Helpers
# =============================================================================
def _expand_targets_for_views(targets: torch.Tensor, num_views: int) -> torch.Tensor:
    return targets.unsqueeze(1).expand(-1, num_views, -1).reshape(-1, targets.shape[-1])


def _get_otcha_list(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "otcha_stages" in output and isinstance(output["otcha_stages"], list):
        return output["otcha_stages"]
    if "otcha" not in output:
        return []
    otcha = output["otcha"]
    if isinstance(otcha, dict):
        return [otcha]
    return []


# =============================================================================
# Training
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,            # BCEWithLogitsLoss
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    label_names: Optional[List[str]] = None,
    md_loss: Optional[nn.Module] = None,
    otra_loss: Optional[nn.Module] = None,
    otra_weight: float = 0.0,
    max_grad_norm: Optional[float] = None,
    use_amp: bool = False,
    logger: Optional[Any] = None,
) -> MetricsResult:

    model.train()
    metrics = MultilabelMetricsComputer(label_names=label_names)

    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", leave=False) \
        if TQDM_AVAILABLE else data_loader

    start_time = time.time()

    loss_m_sum = 0.0
    loss_s_sum = 0.0
    loss_md_sum = 0.0
    loss_rep_sum = 0.0
    num_seen = 0

    beta_sum = 0.0
    gate_sum = 0.0
    confv_sum = 0.0
    otcha_batches = 0

    for images, target, _ in pbar:
        B, N = images.shape[:2]
        images = images.to(device, non_blocking=True)
        target = target.float().to(device, non_blocking=True)  # [B, 14]

        with torch.amp.autocast(device.type, enabled=use_amp):
            output = model(images)
            batch_loss = torch.zeros((), device=device)
            otcha_list = _get_otcha_list(output)

            loss_m = torch.zeros((), device=device)
            logits_mv = None
            if "mv_collection" in output:
                logits_mv = output["mv_collection"]["logits"]  # [B, 14]
                loss_m = criterion(logits_mv, target)
                batch_loss = batch_loss + loss_m

            loss_s = torch.zeros((), device=device)
            if "single" in output:
                logits_single = output["single"]["logits"]          # [B*N, 14]
                target_expanded = _expand_targets_for_views(target, N)  # [B*N, 14]
                loss_s = criterion(logits_single, target_expanded)
                batch_loss = batch_loss + loss_s

            loss_md = torch.zeros((), device=device)
            if md_loss is not None and "single" in output and "mv_collection" in output:
                single_logits = einops.rearrange(output["single"]["logits"], "(b n) k -> b n k", b=B, n=N)
                loss_md = md_loss(output["mv_collection"]["logits"], single_logits, target)
                batch_loss = batch_loss + loss_md

            # --- Lrep: representation KD ---
            loss_rep = torch.zeros((), device=device)
            if otra_loss is not None and otra_weight > 0:
                loss_rep = otra_loss(output)
                batch_loss += otra_weight * loss_rep

        optimizer.zero_grad()
        batch_loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # --- metrics update ---
        if logits_mv is not None:
            metrics.update(logits_mv.detach(), target, batch_loss.detach().item())

        loss_m_sum += loss_m.detach().item() * B
        loss_s_sum += loss_s.detach().item() * B
        loss_md_sum += loss_md.detach().item() * B
        loss_rep_sum += loss_rep.detach().item() * B
        num_seen += B

        # --- OTCHA debug stats ---
        if len(otcha_list) > 0:
            try:
                beta_vals = [s["beta"].detach().mean().item() for s in otcha_list]
                gate_vals = [s["gate"].detach().mean().item() for s in otcha_list]
                confv_vals = [s["conf_v"].detach().mean().item() for s in otcha_list]
                beta_sum += sum(beta_vals) / len(beta_vals)
                gate_sum += sum(gate_vals) / len(gate_vals)
                confv_sum += sum(confv_vals) / len(confv_vals)
                otcha_batches += 1
            except Exception:
                pass

        if TQDM_AVAILABLE:
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    elapsed = time.time() - start_time
    result = metrics.compute()
    result.extra["epoch_time"] = elapsed

    if num_seen > 0:
        result.extra["loss_m"] = loss_m_sum / num_seen
        result.extra["loss_s"] = loss_s_sum / num_seen
        result.extra["loss_md"] = loss_md_sum / num_seen
        result.extra["loss_rep"] = loss_rep_sum / num_seen
        result.extra["otra_weight"] = float(otra_weight)

    if otcha_batches > 0:
        result.extra["beta_mean"] = beta_sum / otcha_batches
        result.extra["gate_mean"] = gate_sum / otcha_batches
        result.extra["confv_mean"] = confv_sum / otcha_batches

    if logger is not None:
        logger.info(f"Epoch {epoch} [Train] {result} | Time: {elapsed:.1f}s")
        if num_seen > 0:
            logger.info(
                f"  Loss: Lm={result.extra.get('loss_m', 0):.4f}, "
                f"Ls={result.extra.get('loss_s', 0):.4f}, "
                f"Lmd={result.extra.get('loss_md', 0):.4f}, "
                f"γ·Lrep={otra_weight * result.extra.get('loss_rep', 0):.4f}"
            )
        if otcha_batches > 0:
            logger.info(
                f"  OTCHA: β={result.extra.get('beta_mean', 0):.4f}, "
                f"gate={result.extra.get('gate_mean', 0):.4f}"
            )

    return result


# =============================================================================
# Validation
# =============================================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    label_names: Optional[List[str]] = None,
    epoch: Optional[int] = None,
    use_amp: bool = False,
    logger: Optional[Any] = None,
) -> MetricsResult:

    model.eval()
    metrics = MultilabelMetricsComputer(label_names=label_names)

    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    pbar = tqdm(data_loader, desc=desc, leave=False) if TQDM_AVAILABLE else data_loader

    start_time = time.time()

    for images, target, _ in pbar:
        images = images.to(device, non_blocking=True)
        target = target.float().to(device, non_blocking=True)  # [B, 14]

        with torch.amp.autocast(device.type, enabled=use_amp):
            output = model(images)

        if "mv_collection" in output:
            logits = output["mv_collection"]["logits"]
            loss = criterion(logits, target)
            metrics.update(logits, target, loss.item())

            if TQDM_AVAILABLE:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    elapsed = time.time() - start_time
    result = metrics.compute()
    result.extra["epoch_time"] = elapsed

    if logger is not None:
        tag = f"Epoch {epoch} " if epoch is not None else ""
        logger.info(f"{tag}[Val] {result} | Time: {elapsed:.1f}s")

    return result


# =============================================================================
# Test
# =============================================================================
@torch.no_grad()
def test(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    criterion: Optional[nn.Module] = None,
    label_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    return_predictions: bool = False,
    use_amp: bool = False,
    logger: Optional[Any] = None,
) -> MetricsResult:

    model.eval()
    metrics = MultilabelMetricsComputer(label_names=label_names, threshold=threshold)

    all_probs = []
    all_preds = []
    all_target = []

    pbar = tqdm(data_loader, desc="Testing", leave=False) if TQDM_AVAILABLE else data_loader

    start_time = time.time()

    for images, target, _ in pbar:
        images = images.to(device, non_blocking=True)
        target = target.float().to(device, non_blocking=True)

        with torch.amp.autocast(device.type, enabled=use_amp):
            output = model(images)

        if "mv_collection" in output:
            logits = output["mv_collection"]["logits"]

            loss_val = 0.0
            if criterion is not None:
                loss_val = criterion(logits, target).item()

            metrics.update(logits, target, loss_val)

            if return_predictions:
                with torch.no_grad():
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= threshold).astype(int)
                all_probs.append(probs)
                all_preds.append(preds)
                all_target.append(target.cpu().numpy())

            if TQDM_AVAILABLE:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    elapsed = time.time() - start_time
    result = metrics.compute()
    result.extra["epoch_time"] = elapsed

    if logger is not None:
        logger.info(f"[Test] {result} | Time: {elapsed:.1f}s")
        # per-label AUC 테이블
        if result.per_label:
            logger.info("Per-label AUC:")
            for name, m in result.per_label.items():
                logger.info(f"  {name:35s}  AUC={m['auc']:.4f}  AP={m['ap']:.4f}")

    if return_predictions:
        predictions = {
            "probabilities": np.concatenate(all_probs),
            "predictions": np.concatenate(all_preds),
            "targets": np.concatenate(all_target),
        }
        return result, predictions

    return result