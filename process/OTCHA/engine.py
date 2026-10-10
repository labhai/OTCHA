# process/OTCHA/engine.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
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
        precision_score, recall_score, confusion_matrix,
        average_precision_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# Metrics
# =============================================================================
@dataclass
class MetricsResult:
    accuracy: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0       # Sensitivity
    specificity: float = 0.0
    confusion_matrix: Optional["np.ndarray"] = None  # type: ignore[name-defined]

    loss: float = 0.0
    num_samples: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "loss": self.loss,
            "num_samples": self.num_samples,
            **{f"extra/{k}": v for k, v in self.extra.items()},
        }

    def __str__(self) -> str:
        return (
            f"Acc: {self.accuracy:.4f} | "
            f"AUC: {self.auc_roc:.4f} | "
            f"F1: {self.f1:.4f} | "
            f"Sens: {self.recall:.4f} | "
            f"Spec: {self.specificity:.4f} | "
            f"Loss: {self.loss:.4f}"
        )


class MetricsComputer:
    def __init__(self, num_classes: int = 2, threshold: float = 0.5):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for MetricsComputer.")
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    @staticmethod
    def _compute_probs_and_preds(logits: torch.Tensor, num_classes: int, threshold: float = 0.5) -> Tuple["np.ndarray", "np.ndarray"]:
        with torch.no_grad():
            if num_classes == 2:
                if logits.ndim == 1 or logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= threshold).astype(int)
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
        return probs, preds

    def reset(self) -> None:
        self.all_preds: List["np.ndarray"] = [] 
        self.all_probs: List["np.ndarray"] = []
        self.all_target: List["np.ndarray"] = []
        self.loss_sum: float = 0.0
        self.num_samples: int = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor, loss: float = 0.0) -> None:
        probs, preds = self._compute_probs_and_preds(logits, self.num_classes, self.threshold)
        target_np = target.detach().cpu().numpy()
        batch_size = len(target_np)

        self.all_preds.append(preds)
        self.all_probs.append(probs)
        self.all_target.append(target_np)
        self.loss_sum += loss * batch_size
        self.num_samples += batch_size

    def compute(self) -> MetricsResult:
        if self.num_samples == 0:
            return MetricsResult()

        preds = np.concatenate(self.all_preds) 
        probs = np.concatenate(self.all_probs)  
        target = np.concatenate(self.all_target) 

        accuracy = accuracy_score(target, preds)

        if self.num_classes == 2:
            try:
                auc_roc = roc_auc_score(target, probs)
            except ValueError:
                auc_roc = 0.0
            try:
                auc_pr = average_precision_score(target, probs)
            except ValueError:
                auc_pr = 0.0

            f1 = f1_score(target, preds, zero_division=0)
            precision_val = precision_score(target, preds, zero_division=0)
            recall_val = recall_score(target, preds, zero_division=0)

            cm = confusion_matrix(target, preds, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                specificity = 0.0
        else:
            try:
                auc_roc = roc_auc_score(target, probs, multi_class="ovr", average="macro")
            except ValueError:
                auc_roc = 0.0
            auc_pr = 0.0
            f1 = f1_score(target, preds, average="macro", zero_division=0)
            precision_val = precision_score(target, preds, average="macro", zero_division=0)
            recall_val = recall_score(target, preds, average="macro", zero_division=0)
            specificity = 0.0
            cm = confusion_matrix(target, preds)

        return MetricsResult(
            accuracy=accuracy,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            f1=f1,
            precision=precision_val,
            recall=recall_val,
            specificity=specificity,
            confusion_matrix=cm,
            loss=self.loss_sum / self.num_samples if self.num_samples > 0 else 0.0,
            num_samples=self.num_samples,
        )
        

def _expand_targets_for_views(targets: torch.Tensor, num_views: int) -> torch.Tensor:
    return targets.unsqueeze(1).expand(-1, num_views).reshape(-1)


def _get_otcha_list(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "otcha_stages" in output and isinstance(output["otcha_stages"], list):
        return output["otcha_stages"]
    if "otcha" not in output:
        return []
    otcha = output["otcha"]
    if isinstance(otcha, dict):
        return [otcha]
    return []


def _get_mean_beta(otcha_list: List[Dict[str, Any]]) -> Optional[float]:
    betas = []
    for si in otcha_list:
        if "beta" in si:
            betas.append(si["beta"].detach().mean().clamp(0.0, 1.0))
    if not betas:
        return None
    return (sum(betas) / len(betas)).item()

# =============================================================================
# Training (OTCHA)
# =============================================================================
def train_one_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, *, num_classes: int = 2, md_loss: Optional[nn.Module] = None,
                    otra_loss: Optional[nn.Module] = None, otra_weight: float = 0.0, max_grad_norm: Optional[float] = None, use_amp: bool = False, logger: Optional[Any] = None) -> MetricsResult:
    model.train()
    metrics = MetricsComputer(num_classes=num_classes)

    if TQDM_AVAILABLE:
        pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    else:
        pbar = data_loader

    start_time = time.time()

    loss_m_sum = 0.0
    loss_s_sum = 0.0
    loss_md_sum = 0.0
    loss_rep_sum = 0.0
    num_seen = 0

    # optional otcha stats
    beta_sum = 0.0
    gate_sum = 0.0
    confv_sum = 0.0
    otcha_batches = 0

    for images, target, _ in pbar:
        B, N = images.shape[:2]
        images = images.to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True).view(-1)

        with torch.amp.autocast(device.type, enabled=use_amp):
            output = model(images)
            batch_loss = torch.zeros((), device=device)

            otcha_list = _get_otcha_list(output)
            
            # --- multi-view classification loss (Lm) ---
            loss_m = torch.zeros((), device=device)
            logits_mv = None
            if "mv_collection" in output:
                logits_mv = output["mv_collection"]["logits"]
                loss_m = criterion(logits_mv, target)
                batch_loss = batch_loss + loss_m

            # --- single-view classification loss (Ls) ---
            loss_s = torch.zeros((), device=device)
            if "single" in output:
                logits_single = output["single"]["logits"]  # (B*N, K)
                target_expanded = _expand_targets_for_views(target, N)  # (B*N,)
                loss_s = criterion(logits_single, target_expanded)
                batch_loss = batch_loss + loss_s

            # --- mutual distillation loss (Lmd) ---
            loss_md = torch.zeros((), device=device)
            if md_loss is not None and "single" in output and "mv_collection" in output:
                single_logits_bn = output["single"]["logits"]  # (B*N, K)
                single_logits = einops.rearrange(single_logits_bn, "(b n) k -> b n k", b=B, n=N)
                loss_md = md_loss(output["mv_collection"]["logits"], single_logits, target)
                batch_loss = batch_loss + loss_md

            # --- representation KD loss (Lrep) ---
            loss_rep = torch.zeros((), device=device)
            if otra_loss is not None and otra_weight > 0:
                loss_rep = otra_loss(output)
                batch_loss += otra_weight * loss_rep

        optimizer.zero_grad()
        batch_loss.backward()
        
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
        optimizer.step()

        if logits_mv is not None:
            metrics.update(logits_mv.detach(), target, float(batch_loss.detach().item()))

        loss_m_sum += float(loss_m.detach().item()) * B
        loss_s_sum += float(loss_s.detach().item()) * B
        loss_md_sum += float(loss_md.detach().item()) * B
        loss_rep_sum += float(loss_rep.detach().item()) * B
        num_seen += B

        # --- otcha debug stats: average across stages ---
        if len(otcha_list) > 0:
            try:
                beta_vals = [si["beta"].detach().mean().item() for si in otcha_list]
                gate_vals = [si["gate"].detach().mean().item() for si in otcha_list]
                confv_vals = [si["conf_v"].detach().mean().item() for si in otcha_list]
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
                f"  Loss breakdown: "
                f"Lm={result.extra.get('loss_m', 0.0):.4f}, "
                f"Ls={result.extra.get('loss_s', 0.0):.4f}, "
                f"Lmd={result.extra.get('loss_md', 0.0):.4f}, "
                f"gamma*Lrep={float(otra_weight)*result.extra.get('loss_rep', 0.0):.4f}"
            )
        if otcha_batches > 0:
            logger.info(
                f"  OTCHA stats: beta_mean={result.extra.get('beta_mean', 0.0):.4f}, "
                f"gate_mean={result.extra.get('gate_mean', 0.0):.4f}"
            )

    return result


# =============================================================================
# Validation
# =============================================================================
@torch.no_grad()
def validate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int = 2, *, epoch: Optional[int] = None, use_amp: bool = False, logger: Optional[Any] = None) -> MetricsResult:
    model.eval()
    metrics = MetricsComputer(num_classes=num_classes)

    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    if TQDM_AVAILABLE:
        pbar = tqdm(data_loader, desc=desc, leave=False)
    else:
        pbar = data_loader

    start_time = time.time()

    for images, target, _ in pbar:
        images = images.to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True).view(-1)

        with torch.amp.autocast(device.type, enabled=use_amp):
            output = model(images)

        if "mv_collection" in output:
            logits = output["mv_collection"]["logits"]
            loss = criterion(logits, target)
            metrics.update(logits, target, float(loss.item()))

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
def test(model: nn.Module, data_loader: DataLoader, device: torch.device, *, criterion: Optional[nn.Module] = None, num_classes: int = 2, threshold: float = 0.5, return_predictions: bool = False, use_amp: bool = False, logger: Optional[Any] = None):
    model.eval()
    metrics = MetricsComputer(num_classes=num_classes, threshold=threshold)

    all_preds = []
    all_probs = []
    all_target = []

    if TQDM_AVAILABLE:
        pbar = tqdm(data_loader, desc="Testing", leave=False)
    else:
        pbar = data_loader

    start_time = time.time()

    for images, target, _ in pbar:
        images = images.to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True).view(-1)

        with torch.amp.autocast(device.type, enabled=use_amp):
            output = model(images)

        if "mv_collection" in output:
            logits = output["mv_collection"]["logits"]

            if criterion is not None:
                loss = criterion(logits, target)
                loss_val = float(loss.item())
            else:
                loss_val = 0.0

            metrics.update(logits, target, loss_val)

            if return_predictions:
                probs, preds = MetricsComputer._compute_probs_and_preds(logits, num_classes, threshold)
                all_preds.append(preds)
                all_probs.append(probs)
                all_target.append(target.detach().cpu().numpy())

            if TQDM_AVAILABLE:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    elapsed = time.time() - start_time
    result = metrics.compute()
    result.extra["epoch_time"] = elapsed

    if logger is not None:
        logger.info(f"[Test] {result} | Time: {elapsed:.1f}s")
        logger.info(f"Confusion Matrix:\n{result.confusion_matrix}")

    if return_predictions:
        predictions = {
            "predictions": np.concatenate(all_preds),
            "probabilities": np.concatenate(all_probs),
            "targets": np.concatenate(all_target),
        }
        return result, predictions

    return result
