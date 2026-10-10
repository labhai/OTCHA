# CheXpert_otcha.py
from __future__ import annotations

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Dataset (CheXpert)
from dataset.chexpert import CheXpertDataset, CHEXPERT_LABELS, NUM_LABELS

# Network
from models.OTCHA.otcha import OTCHA

# Process (CheXpert-dedicated engine)
from process.OTCHA.chexpert_engine import train_one_epoch, validate, test

# Loss
from loss.otra_loss import OTRALoss
from loss.chexpert_md_loss import MutualDistillationLoss

# Utils
from utils import save_results_csv, EarlyStopper, set_seed, set_optimizer, str2bool, create_scheduler

def parse_option():
    parser = argparse.ArgumentParser(description="OTCHA Training on CheXpert")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_view", type=int, default=2, help="frontal + lateral")
    parser.add_argument("--num_classes", type=int, default=NUM_LABELS, help="13 labels (multilabel)")
    parser.add_argument("--dataset", type=str, default="chexpert")

    # Method & Model
    parser.add_argument("--method", type=str, default="OTCHA")
    parser.add_argument("--model_name", type=str, default="vit_small_r26_s32_224")
    parser.add_argument("--pretrained_weights", type=str2bool, default=True)

    # Fused OT (pos term)
    parser.add_argument("--num_latents", type=int, default=49)
    parser.add_argument("--latent_refine_iters", type=int, default=1)
    parser.add_argument("--ot_pos_weight", type=float, default=0.2)
    parser.add_argument("--ot_pos_sigma", type=float, default=1.0)

    # OTCHA params
    parser.add_argument("--sinkhorn_iters", type=int, default=5)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.1)
    parser.add_argument("--bin_score_init", type=float, default=0.0)
    parser.add_argument("--normalize_qk", type=str2bool, default=True)
    parser.add_argument("--refine_dropout", type=float, default=0.0)
    parser.add_argument("--detach_transport", type=str2bool, default=False)
    parser.add_argument("--return_transport_matrix", type=str2bool, default=True)
    parser.add_argument("--num_otcha_stages", type=int, default=0)

    # Loss toggles
    parser.add_argument("--md_loss", type=str2bool, default=True)
    parser.add_argument("--otra_loss", type=str2bool, default=True)

    # MD hyperparameters
    parser.add_argument("--temp", type=float, default=4.0)
    parser.add_argument("--lambda_param", type=float, default=0.1)

    # OTRA hyperparameters
    parser.add_argument("--otra_weight", type=float, default=0.1)
    parser.add_argument("--otra_student_key", type=str, default="tokens_refined", choices=["tokens", "tokens_refined"])
    parser.add_argument("--otra_weight_mode", type=str, default="gate_beta", choices=["gate_beta", "gate_beta_confv", "confv_beta", "gate", "beta", "uniform"])
    parser.add_argument("--otra_detach_teacher", type=str2bool, default=True)
    parser.add_argument("--otra_detach_weights", type=str2bool, default=True)
    parser.add_argument("--otra_feature_norm", type=str, default="layernorm+l2", choices=["layernorm+l2", "l2", "none"])
    parser.add_argument("--otra_distance", type=str, default="cosine", choices=["cosine", "mse", "smooth_l1"])
    parser.add_argument("--otra_debug_checks", type=str2bool, default=False)
    parser.add_argument("--use_stage", type=str2bool, default=False)
    parser.add_argument("--stage_start", type=int, default=0, choices=[0, 1])
    parser.add_argument("--stage_weight", type=float, default=1.0)

    # Data paths (CheXpert)
    parser.add_argument("--data_dir", type=str, default="data/CheXpert", help="CheXpert data directory")
    parser.add_argument("--csv_file", type=str, default="chexpert_label.csv")
    parser.add_argument("--splits_json", type=str, default="splits.json")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--expand_channels", type=int, default=3, help="1: grayscale, 3: expand to RGB")
    parser.add_argument("--use_clahe", type=str2bool, default=False)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip_norm", type=float, default=80.0)

    # AMP
    parser.add_argument("--use_amp", type=str2bool, default=False)

    # Optimization
    parser.add_argument("--optim", type=str, default="SGD", choices=["Adam", "AdamW", "SGD"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["none", "cosinewarmup", "cosine", "step", "plateau"])
    parser.add_argument("--lr_min", type=float, default=1e-8)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--pl_patience", type=int, default=10)

    opt = parser.parse_args()
    return opt

# =============================================================================
# Dataset
# =============================================================================
def dataset_loader(opt):
    train_dataset = CheXpertDataset(
        data_dir=opt.data_dir, csv_file=opt.csv_file, splits_json=opt.splits_json,
        split="train", image_size=opt.image_size,
        expand_channels=opt.expand_channels, use_clahe=opt.use_clahe,
    )
    val_dataset = CheXpertDataset(
        data_dir=opt.data_dir, csv_file=opt.csv_file, splits_json=opt.splits_json,
        split="val", image_size=opt.image_size,
        expand_channels=opt.expand_channels, use_clahe=opt.use_clahe,
    )
    test_dataset = CheXpertDataset(
        data_dir=opt.data_dir, csv_file=opt.csv_file, splits_json=opt.splits_json,
        split="test", image_size=opt.image_size,
        expand_channels=opt.expand_channels, use_clahe=opt.use_clahe,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True, num_workers=opt.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.num_workers,
        drop_last=False, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.num_workers,
        drop_last=False, pin_memory=True,
    )

    print(f"CheXpert loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    return train_loader, val_loader, test_loader

# =============================================================================
# Model & Losses
# =============================================================================
def set_model_and_losses(opt, device):
    model = OTCHA(
        opt=opt,
        model_name=opt.model_name,
        num_classes=opt.num_classes,       # 14
        n=opt.num_view,                    # 2 (frontal + lateral)
        pretrained_weights=opt.pretrained_weights,
        sinkhorn_iters=opt.sinkhorn_iters,
        sinkhorn_epsilon=opt.sinkhorn_epsilon,
        bin_score_init=opt.bin_score_init,
        normalize_qk=opt.normalize_qk,
        refine_dropout=opt.refine_dropout,
        detach_transport=opt.detach_transport,
        ot_pos_weight=opt.ot_pos_weight,
        ot_pos_sigma=opt.ot_pos_sigma,
        return_otcha_info=True,
        return_transport_matrix=opt.return_transport_matrix,
        num_otcha_stages=opt.num_otcha_stages,
    ).to(device)

    # CheXpert: BCEWithLogitsLoss (multilabel)
    criterion = nn.BCEWithLogitsLoss()

    # Mutual Distillation: sigmoid binary KL (CheXpert 전용)
    md_loss_fn = None
    if opt.md_loss:
        md_loss_fn = MutualDistillationLoss(temp=opt.temp, lambda_hyperparam=opt.lambda_param).to(device)

    otra_loss_fn = None
    if opt.otra_loss:
        otra_loss_fn = OTRALoss(
            student_key=opt.otra_student_key,
            weight_mode=opt.otra_weight_mode,
            detach_teacher=opt.otra_detach_teacher,
            detach_weights=opt.otra_detach_weights,
            feature_norm=opt.otra_feature_norm,
            distance=opt.otra_distance,
            debug_checks=opt.otra_debug_checks,
            use_otcha_stages=opt.use_stage,
            stage_start=opt.stage_start,
            stage_weight_decay=opt.stage_weight,
        ).to(device)

    return model, criterion, md_loss_fn, otra_loss_fn

# =============================================================================
# Scheduler
# =============================================================================
def setup_scheduler(opt, optimizer):
    return create_scheduler(opt, optimizer)

# =============================================================================
# Training Loop
# =============================================================================
def train_process(opt, train_loader, val_loader, model, criterion, md_loss, otra_loss, optimizer, scheduler, device):
    save_dir = f"result/{opt.method}/{opt.dataset}/save_pth"
    os.makedirs(save_dir, exist_ok=True)

    save_path = (
        f"{save_dir}/"
        f"{opt.model_name}_seed{opt.seed}_n{opt.num_view}_bs{opt.batch_size}_"
        f"{opt.optim}_lr{opt.lr}_"
        f"stages{opt.num_otcha_stages}_eps{opt.sinkhorn_epsilon}_"
        f"M{opt.num_latents}_bin{opt.bin_score_init}_posW{opt.ot_pos_weight}_posS{opt.ot_pos_sigma}_"
        f"md({opt.md_loss})_lam{opt.lambda_param}_"
        f"otra({opt.otra_loss})_gamma{opt.otra_weight}_{opt.otra_weight_mode}_"
        f"normqk({opt.normalize_qk})_detP({opt.detach_transport}).pth"
    )

    early_stopper = EarlyStopper(patience=opt.patience, use_loss_when_tie=True, save_path=save_path)

    for epoch in range(1, opt.epochs + 1):
        start_time = time.time()

        # ---- Train (engine_chexpert) ----
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            label_names=CHEXPERT_LABELS,
            md_loss=md_loss,
            otra_loss=otra_loss,
            otra_weight=opt.otra_weight,
            max_grad_norm=opt.grad_clip_norm,
            use_amp=opt.use_amp,
        )

        # ---- Validate (engine_chexpert) ----
        val_result = validate(
            model, val_loader, criterion, device,
            label_names=CHEXPERT_LABELS,
            epoch=epoch,
            use_amp=opt.use_amp,
        )

        # ---- Scheduler step ----
        if scheduler is not None:
            if opt.scheduler == "plateau":
                scheduler.step(val_result.mean_auc)
            else:
                scheduler.step()

        end_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{opt.epochs} | "
            f"Tr Loss: {train_result.loss:.4f}, Tr mAUC: {train_result.mean_auc:.4f} | "
            f"Val Loss: {val_result.loss:.4f}, Val mAUC: {val_result.mean_auc:.4f}, "
            f"Val mF1: {val_result.mean_f1:.4f} | "
            f"lr: {current_lr:.6f}, Time: {end_time - start_time:.1f}s"
        )

        # ---- Early stopping (monitor val mean AUC) ----
        val_auc = val_result.mean_auc
        val_loss = val_result.loss
        best_improved, should_stop = early_stopper.check(metric=val_auc, model=model, tiebreak_loss=val_loss)

        if best_improved:
            print(f"  -> New best model saved! (mAUC: {val_auc:.4f})")

        if should_stop:
            print(f"  -> Early stopping triggered at epoch {epoch}.")
            break

    return save_path

# =============================================================================
# Main
# =============================================================================
def main():
    opt = parse_option()
    set_seed(opt.seed)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    # ---- Dataloader ----
    train_loader, val_loader, test_loader = dataset_loader(opt)

    # ---- Model / Losses ----
    model, criterion, md_loss, otra_loss = set_model_and_losses(opt, device)

    # ---- Optimizer / Scheduler ----
    optimizer = set_optimizer(opt, model)
    scheduler = setup_scheduler(opt, optimizer)

    # === Training ===
    print(f"\nStart Training: {opt.method} on {opt.dataset}")
    print(f"  Labels: {NUM_LABELS}, Views: {opt.num_view}, Criterion: BCEWithLogitsLoss")
    print(f"  MD Loss: {opt.md_loss}, OTRA Loss: {opt.otra_loss}\n")

    best_model_path = train_process(
        opt, train_loader, val_loader,
        model, criterion, md_loss, otra_loss,
        optimizer, scheduler, device,
    )

    # === Testing ===
    print("\nStart Testing...")
    if os.path.exists(best_model_path):
        if torch.__version__ >= "2.0":
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        else:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model file not found! Using current model state.")

    test_result = test(
        model, test_loader, device,
        criterion=criterion,
        label_names=CHEXPERT_LABELS,
    )

    # ---- Print test results ----
    print(f"\n{'=' * 60}")
    print("  CheXpert Test Results (Mean over 13 labels)")
    print(f"  mAUC : {test_result.mean_auc:.4f}")
    print(f"  mAP  : {test_result.mean_ap:.4f}")
    print(f"  mAcc : {test_result.mean_acc:.4f}")
    print(f"  mF1  : {test_result.mean_f1:.4f}")
    print(f"  mPrec: {test_result.mean_precision:.4f}")
    print(f"  mRec : {test_result.mean_recall:.4f}")
    print(f"{'=' * 60}")
    print("  Per-label AUC:")
    for label_name, metrics in test_result.per_label.items():
        print(f"    {label_name:35s}  AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}")
    print(f"{'=' * 60}\n")

    # ---- Save results (TXT) ----
    results_dir = f"result/{opt.method}/txt"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/{opt.dataset}_{opt.model_name}_results_summary.txt"

    with open(results_path, "a") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M')}] {opt.dataset}\n")
        f.write(
            f"Config: Seed={opt.seed}, Model={opt.model_name}, N={opt.num_view}, "
            f"BS={opt.batch_size}, LR={opt.lr}, Optim={opt.optim}, "
            f"MD={opt.md_loss}(tau={opt.temp}, lam={opt.lambda_param}), "
            f"otra={opt.otra_loss}(gamma={opt.otra_weight}, mode={opt.otra_weight_mode})\n"
        )
        f.write(f"  mAUC : {test_result.mean_auc:.4f}\n")
        f.write(f"  mAP  : {test_result.mean_ap:.4f}\n")
        f.write(f"  mAcc : {test_result.mean_acc:.4f}\n")
        f.write(f"  mF1  : {test_result.mean_f1:.4f}\n")
        f.write(f"  mPrec: {test_result.mean_precision:.4f}\n")
        f.write(f"  mRec : {test_result.mean_recall:.4f}\n")
        f.write("  Per-label AUC:\n")
        for label_name, metrics in test_result.per_label.items():
            f.write(f"    {label_name:35s}  {metrics['auc']:.4f}\n")
        f.write("-" * 30 + "\n")

    print(f"Results saved to {results_path}")

    # save_results_csv(opt, test_result)
    print("All processes finished.")

if __name__ == "__main__":
    main()
