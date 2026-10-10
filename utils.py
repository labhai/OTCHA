from __future__ import annotations

import argparse
import csv
import math
import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def create_scheduler(args, optimizer):
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_csv(opt, test_result):
    csv_dir = f"result/{opt.method}/csv"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{opt.dataset}_results.csv")

    columns = [
        "model_name", "seed", "num_view", "batch_size", "optim", "lr",
        "weight_decay", "num_latents", "ot_pos_weight", "sinkhorn_epsilon", "sinkhorn_iters",
        "md_loss", "otra_loss", "lambda_param", "otra_weight",
        "AUC", "F1", "Acc", "Sens", "Spec",
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
