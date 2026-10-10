from __future__ import annotations

import os
import csv
import random
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from torch.cuda.amp import GradScaler

import torch
import math
import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, StepLR,
)


def create_scheduler(args, optimizer):
    """Create a PyTorch scheduler; warmup/cosine cycles need no extra package."""
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosinewarmup":
        if args.lr <= 0 or not 0 <= args.lr_min <= args.lr:
            raise ValueError("cosinewarmup requires 0 <= lr_min <= lr and lr > 0")
        cycle_length = max(1, args.epochs // 2)
        warmup_steps = min(5, cycle_length - 1)

        def lr_factor(epoch):
            cycle, position = divmod(epoch, cycle_length)
            peak_lr = args.lr * (0.5 ** cycle)
            if position < warmup_steps:
                lr = args.lr_min + (peak_lr - args.lr_min) * position / warmup_steps
            else:
                progress = (position - warmup_steps) / (cycle_length - warmup_steps)
                lr = args.lr_min + (peak_lr - args.lr_min) * (1 + math.cos(math.pi * progress)) / 2
            return lr / args.lr

        return LambdaLR(optimizer, lr_lambda=lr_factor)
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    if args.scheduler == "step":
        return StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=args.pl_patience, min_lr=args.lr_min,
        )
    raise ValueError(f"Unknown scheduler: {args.scheduler}")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def k_fold_save_results_csv(opt, summary):
    csv_dir = f"result/{opt.method}/csv"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{opt.dataset}_results.csv")

    columns = [
        "model_name", "seed", "num_view", "batch_size", "optim", "lr",
        "weight_decay", "num_latents", "ot_pos_weight", "sinkhorn_epsilon",
        "md_loss", "otra_loss", "lambda_param", "otra_weight",
        "detach_transport", "AUC", "F1", "Acc", "Sens", "Spec",
    ]

    row = {
        "model_name": opt.model_name,
        "seed": opt.seed,
        "num_view": opt.num_view,
        "batch_size": opt.batch_size,
        "optim": opt.optim,
        "lr": opt.lr,
        "weight_decay": opt.weight_decay,
        "num_latents": opt.num_latents,
        "ot_pos_weight": opt.ot_pos_weight,
        "sinkhorn_epsilon": opt.sinkhorn_epsilon,
        "md_loss": opt.md_loss,
        "otra_loss": opt.otra_loss,
        "lambda_param": opt.lambda_param,
        "otra_weight": opt.otra_weight,
        "detach_transport": opt.detach_transport,
        "AUC": f"{summary['auc_roc']['mean']:.4f}±{summary['auc_roc']['std']:.4f}",
        "F1": f"{summary['f1']['mean']:.4f}±{summary['f1']['std']:.4f}",
        "Acc": f"{summary['accuracy']['mean']:.2f}±{summary['accuracy']['std']:.2f}",
        "Sens": f"{summary['recall']['mean']:.4f}±{summary['recall']['std']:.4f}",
        "Spec": f"{summary['specificity']['mean']:.4f}±{summary['specificity']['std']:.4f}",
    }

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Results appended to {csv_path}")
    
def save_results_csv(opt, test_result):
    csv_dir = f"result/{opt.method}/csv"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{opt.dataset}_results.csv")

    columns = [
        "model_name", "seed", "num_view", "batch_size", "optim", "lr",
        "weight_decay", "num_latents", "ot_pos_weight", "sinkhorn_epsilon", "sinkhorn_iters",
        "md_loss", "otra_loss", "lambda_param", "otra_weight",
        "detach_transport", "AUC", "F1", "Acc", "Sens", "Spec",
    ]

    row = {
        "model_name": opt.model_name,
        "seed": opt.seed,
        "num_view": opt.num_view,
        "batch_size": opt.batch_size,
        "optim": opt.optim,
        "lr": opt.lr,
        "weight_decay": opt.weight_decay,
        "num_latents": opt.num_latents,
        "ot_pos_weight": opt.ot_pos_weight,
        "sinkhorn_epsilon": opt.sinkhorn_epsilon,
        "sinkhorn_iters": opt.sinkhorn_iters,
        "md_loss": opt.md_loss,
        "otra_loss": opt.otra_loss,
        "lambda_param": opt.lambda_param,
        "otra_weight": opt.otra_weight,
        "detach_transport": opt.detach_transport,
        "AUC": round(test_result.auc_roc, 4),
        "F1": round(test_result.f1, 4),
        "Acc": round(test_result.accuracy * 100, 2),
        "Sens": round(test_result.recall, 4),
        "Spec": round(test_result.specificity, 4),
    }

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Results appended to {csv_path}")
    
    
def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_scaler(use_amp: bool = False) -> Optional[GradScaler]:
    if use_amp:
        return torch.amp.GradScaler('cuda')
    return None


class EarlyStopper:
    def __init__(self, patience=5, use_loss_when_tie=True, save_path='best_model.pth', eps=1e-12):
        self.patience = patience
        self.use_loss_when_tie = use_loss_when_tie
        self.save_path = save_path
        self.eps = eps

        self.best_metric = -math.inf
        self.best_tiebreak_loss = math.inf

        self.best_stop_metric = -math.inf
        self.stop_counter = 0

    def check(self, metric, model, tiebreak_loss=None):
        saved = False

        better = metric > self.best_metric + self.eps
        tie = abs(metric - self.best_metric) <= self.eps

        if better:
            saved = True
            self.best_metric = metric
            if tiebreak_loss is not None:
                self.best_tiebreak_loss = tiebreak_loss

        elif tie and self.use_loss_when_tie and (tiebreak_loss is not None):
            if tiebreak_loss < self.best_tiebreak_loss - self.eps:
                saved = True
                self.best_tiebreak_loss = tiebreak_loss

        if saved:
            self._save_checkpoint(model)

        improved = metric > self.best_stop_metric + self.eps
        if improved:
            self.best_stop_metric = metric
            self.stop_counter = 0
        else:
            self.stop_counter += 1
            print(f"No improvement for {self.stop_counter} epoch(s)")

        stopped = self.stop_counter >= self.patience
        return saved, stopped

    def _save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"New best model saved: {self.save_path}")
        
        
# =========================================================================================================== #
#                                            Setting optimizer                                                #
# =========================================================================================================== # 
def set_optimizer(args, model):
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=getattr(args, "momentum", 0.9), weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    else:
        raise ValueError(f"Unknown optimizer: {args.optim}")
        
    return optimizer

# =============================================================================
# Checkpoint utilities
# =============================================================================

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, results: Dict[str, float], filepath: str, scheduler: Optional[Any] = None, 
                    scaler: Optional[GradScaler] = None, fdbn_epoch: int = 0, is_best: bool = False, best_filepath: Optional[str] = None,):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'fdbn_epoch': fdbn_epoch,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    if is_best:
        import shutil
        if best_filepath is None:
            best_filepath = filepath.replace('.pth', '_best.pth')
        shutil.copy(filepath, best_filepath)
        print(f"Best model saved: {best_filepath}")


def load_checkpoint(filepath: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None, scaler: Optional[GradScaler] = None, device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model weights loaded from: {filepath}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'results': checkpoint.get('results', {}),
        'fdbn_epoch': checkpoint.get('fdbn_epoch', 0),
    }
    
# =============================================================================
# Logging utilities
# =============================================================================

def log_metrics(results: Dict[str, float], prefix: str = "", epoch: Optional[int] = None):
    header = f"[Epoch {epoch}] " if epoch is not None else ""
    header += f"{prefix}" if prefix else ""

    # Loss
    loss_str = f"Loss: {results.get('loss_total', 0):.4f}"
    if 'loss_cls' in results:
        loss_str += (
            f" (cls: {results.get('loss_cls', 0):.4f}, "
            f"cycle: {results.get('loss_cycle', 0):.4f}, "
            f"contrast: {results.get('loss_contrast', 0):.4f})"
        )

    cls_str = ""
    if ('auc_roc' in results) or ('accuracy' in results):
        cls_str = (
            f"AUC: {results.get('auc_roc', 0):.4f}, "
            f"AP: {results.get('auc_pr', 0):.4f}, "
            f"Acc: {results.get('accuracy', 0):.4f}, "
            f"Sen: {results.get('recall', 0):.4f}, "
            f"Spe: {results.get('specificity', 0):.4f}, "
            f"F1: {results.get('f1', 0):.4f}"
        )

    print(f"{header} {loss_str}")
    if cls_str:
        print(f"  Classification: {cls_str}")


def format_metrics_oneline(results: Dict[str, float]) -> str:
    return (
        f"Loss: {results.get('loss_total', 0):.4f} | "
        f"AUC: {results.get('auc_roc', 0):.4f} | "
        f"Acc: {results.get('accuracy', 0):.4f} | "
        f"F1: {results.get('f1', 0):.4f}"
    )
    
def format_metrics(metrics) -> str:
    if hasattr(metrics, 'to_dict'):
        m = metrics.to_dict()
    else:
        m = metrics
    
    parts = [f"Loss={m['loss_total']:.4f}"]
    
    if 'accuracy' in m:
        parts.append(f"Acc={m['accuracy']*100:.2f}%")
    
    auc_val = m.get('auc_roc') or m.get('auc')
    if auc_val is not None:
        parts.append(f"AUC={auc_val:.4f}")
    
    if 'f1' in m:
        parts.append(f"F1={m['f1']:.4f}")
    
    if 'recall' in m:
        parts.append(f"Sens={m['recall']:.4f}")
    
    if 'specificity' in m:
        parts.append(f"Spec={m['specificity']:.4f}")
    
    return " | ".join(parts)
