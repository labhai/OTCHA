# main_otcha.py
from __future__ import annotations

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Dataset (VinDr-Mammo)
from dataset.vindr_mammo import VinDrDataset

# Network
from models.OTCHA.otcha import OTCHA

# Process
from process.OTCHA.engine import train_one_epoch, validate, test

# Loss
from loss.otra_loss import OTRALoss, MutualDistillationLoss

# Utils
from utils import save_results_csv, EarlyStopper, set_seed, set_optimizer, str2bool, create_scheduler

def parse_option():
    parser = argparse.ArgumentParser(description="OTCHA Training on VinDr-Mammo")

    parser.add_argument("--device", type=int, default=0, help="GPU number")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_view", type=int, default=2, help="number of views (2 or 4)")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--dataset", type=str, default="vindr", help="dataset name (currently vindr)")

    # Method & Model
    parser.add_argument("--method", type=str, default="OTCHA", help="method name")
    parser.add_argument("--model_name", type=str, default="vit_small_r26_s32_224", help="timm model name")
    parser.add_argument("--pretrained_weights", type=str2bool, default=True, help="use timm pretrained weights")

    # fused OT (pos term)
    parser.add_argument("--num_latents", type=int, default=64, help="num latents M")
    parser.add_argument("--latent_refine_iters", type=int, default=1, help="latent aggregation iterations")
    parser.add_argument("--ot_pos_weight", type=float, default=0.2, help="positive geometry weight for fused OT; must be > 0")
    parser.add_argument("--ot_pos_sigma", type=float, default=1.0, help="sigma for Gaussian-like positional score: -d^2/(2*sigma^2)")
    
    # OTCHA params 
    parser.add_argument("--sinkhorn_iters", type=int, default=5, help="sinkhorn iterations")
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.1, help="sinkhorn epsilon")
    parser.add_argument("--bin_score_init", type=float, default=0.0, help="dustbin logit init")
    parser.add_argument("--normalize_qk", type=str2bool, default=True, help="L2 normalize q/k for cosine scores")
    parser.add_argument("--refine_dropout", type=float, default=0.0, help="dropout on aggregated message")
    parser.add_argument("--detach_transport", type=str2bool, default=False, help="detach OT matrix P (ablation)")
    parser.add_argument("--return_transport_matrix", type=str2bool, default=True, help="store P in otcha_info (debug)")
    parser.add_argument("--num_otcha_stages", type=int, default=0, help="number of mid-layer OTCHA stages (0 disables)")

    # Loss toggles
    parser.add_argument("--md_loss", type=str2bool, default=True, help="use mutual distillation loss")
    parser.add_argument("--otra_loss", type=str2bool, default=True, help="use OTCHA rep-kd loss")

    # MD hyperparameters
    parser.add_argument("--temp", type=float, default=4.0, help="mutual distillation temperature")
    parser.add_argument("--lambda_param", type=float, default=0.1, help="mutual distillation lambda (inside md_loss)")

    # otra hyperparameters
    parser.add_argument("--otra_weight", type=float, default=0.1, help="gamma: weight for otra loss")
    parser.add_argument("--otra_student_key", type=str, default="tokens_refined", choices=["tokens", "tokens_refined"], help="student tokens key for otra")
    parser.add_argument("--otra_weight_mode", type=str, default="gate_beta", choices=["gate_beta", "gate_beta_confv", "confv_beta", "gate", "beta", "uniform"], help="otra weight mode based on otcha_info")
    parser.add_argument("--otra_detach_teacher", type=str2bool, default=True, help="detach teacher(message) in otra")
    parser.add_argument("--otra_detach_weights", type=str2bool, default=True, help="detach weights(gate/beta/conf) in otra")
    parser.add_argument("--otra_feature_norm", type=str, default="layernorm+l2", choices=["layernorm+l2", "l2", "none"], help="feature normalization for otra")
    parser.add_argument("--otra_distance", type=str, default="cosine", choices=["cosine", "mse", "smooth_l1"], help="distance metric for otra")
    parser.add_argument("--otra_debug_checks", type=str2bool, default=False, help="enable shape checks in otra")
    parser.add_argument("--use_stage", type=str2bool, default=False, help="use or not use stage-wise otra Loss")
    parser.add_argument("--stage_start", type=int, default=0, choices=[0, 1], help="0: include stage0, 1: mid-stages only")
    parser.add_argument("--stage_weight", type=float, default=1.0, help="stage weight for otra Loss")
    
    # Data paths (VinDr)
    parser.add_argument("--data_dir", type=str, default="data/VinDr_Mammo", help="VinDr-Mammo data directory")
    parser.add_argument("--csv_file", type=str, default="breast-level_annotations.csv", help="VinDr meta csv file")
    parser.add_argument("--splits_json", type=str, default="splits.json", help="splits json")
    parser.add_argument("--data_mode", type=str, default="2view", help="2view or 4view")
    parser.add_argument("--image_size", type=int, default=224, help="input image size")
    parser.add_argument("--use_clahe", type=str2bool, default=False, help="apply CLAHE")

    # Sampler
    parser.add_argument("--use_sampler", type=str2bool, default=False, help="use WeightedRandomSampler")
    parser.add_argument("--sampler_pos_fraction", type=float, default=0.25, help="target positive fraction")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--grad_clip_norm", type=float, default=80.0, help="grad clip norm")

    # AMP
    parser.add_argument("--use_amp", type=str2bool, default=False, help="use torch.amp autocast + GradScaler")

    # Optimization
    parser.add_argument("--optim", type=str, default="SGD", choices=["Adam", "AdamW", "SGD"], help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["none", "cosinewarmup", "cosine", "step", "plateau"], help="lr scheduler")
    parser.add_argument("--lr_min", type=float, default=1e-8, help="minimum lr")
    parser.add_argument("--step_size", type=int, default=20, help="StepLR step size")
    parser.add_argument("--pl_patience", type=int, default=10, help="plateau patience")

    opt = parser.parse_args()
    return opt

def dataset_loader(opt):
    if opt.dataset != "vindr":
        raise ValueError(f"Unknown dataset: {opt.dataset} (this script currently supports only 'vindr').")

    train_dataset = VinDrDataset(
        data_dir=opt.data_dir, csv_file=opt.csv_file, splits_json=opt.splits_json,
        split="train", mode=opt.data_mode, image_size=opt.image_size, expand_channels=3, use_clahe=opt.use_clahe
    )
    val_dataset = VinDrDataset(
        data_dir=opt.data_dir, csv_file=opt.csv_file, splits_json=opt.splits_json,
        split="val", mode=opt.data_mode, image_size=opt.image_size, expand_channels=3, use_clahe=opt.use_clahe
    )
    test_dataset = VinDrDataset(
        data_dir=opt.data_dir, csv_file=opt.csv_file, splits_json=opt.splits_json,
        split="test", mode=opt.data_mode, image_size=opt.image_size, expand_channels=3, use_clahe=opt.use_clahe
    )

    # WeightedRandomSampler for class imbalance (optional)
    if opt.use_sampler:
        labels = train_dataset.labels
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())

        p = float(opt.sampler_pos_fraction)
        w_pos = p / max(n_pos, 1)
        w_neg = (1.0 - p) / max(n_neg, 1)

        weights = np.where(labels == 1, w_pos, w_neg)
        weights = torch.as_tensor(weights, dtype=torch.double)

        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size,
        sampler=sampler, shuffle=train_shuffle,
        num_workers=opt.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.num_workers,
        drop_last=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.num_workers,
        drop_last=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader

def set_model_and_losses(opt, device):
    model = OTCHA(
        opt=opt,
        model_name=opt.model_name,
        num_classes=opt.num_classes,
        n=opt.num_view,
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
        num_otcha_stages=opt.num_otcha_stages
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    md_loss = None
    if opt.md_loss:
        md_loss = MutualDistillationLoss(temp=opt.temp, lambda_hyperparam=opt.lambda_param).to(device)

    otra_loss = None
    if opt.otra_loss:
        otra_loss = OTRALoss(
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

    return model, criterion, md_loss, otra_loss

def setup_scheduler(opt, optimizer):
    return create_scheduler(opt, optimizer)

def train_process(opt, train_loader, val_loader, model, criterion, md_loss, otra_loss, optimizer, scheduler, device):
    save_dir = f"result/{opt.method}/{opt.dataset}/save_pth"
    os.makedirs(save_dir, exist_ok=True)

    save_path = (
        f"{save_dir}/"
        f"{opt.model_name}_seed{opt.seed}_n{opt.num_view}_bs{opt.batch_size}_"
        f"{opt.optim}_lr{opt.lr}_"
        f"stages{opt.num_otcha_stages}_eps{opt.sinkhorn_epsilon}_eps_iters{opt.sinkhorn_iters}_"
        f"M{opt.num_latents}_bin{opt.bin_score_init}_posW{opt.ot_pos_weight}_posS{opt.ot_pos_sigma}_"
        f"md({opt.md_loss})_lam{opt.lambda_param}_"
        f"otra({opt.otra_loss})_gamma{opt.otra_weight}_{opt.otra_weight_mode}_"
        f"normqk({opt.normalize_qk})_detP({opt.detach_transport}).pth"
    )

    early_stopper = EarlyStopper(patience=getattr(opt, "patience", 5), use_loss_when_tie=True, save_path=save_path)

    for epoch in range(1, opt.epochs + 1):
        start_time = time.time()

        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_classes=opt.num_classes, md_loss=md_loss,
                                       otra_loss=otra_loss, otra_weight=opt.otra_weight, max_grad_norm=opt.grad_clip_norm, use_amp=opt.use_amp)
        val_result = validate(model, val_loader, criterion, device, num_classes=opt.num_classes, epoch=epoch)

        # scheduler step
        if scheduler is not None:
            if opt.scheduler == "plateau":
                scheduler.step(val_result.auc_roc)
            else:
                scheduler.step()

        end_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{opt.epochs} | "
            f"Tr Loss: {train_result.loss:.4f}, Tr Acc: {train_result.accuracy * 100:.2f}% | "
            f"Val Loss: {val_result.loss:.4f}, Val Acc: {val_result.accuracy * 100:.2f}%, "
            f"Val AUC: {val_result.auc_roc:.4f} | "
            f"lr: {current_lr:.6f}, Time: {end_time - start_time:.1f}s"
        )

        # Early stopping (monitor val AUC)
        val_auc = val_result.auc_roc
        val_loss = val_result.loss
        best_improved, should_stop = early_stopper.check(metric=val_auc, model=model, tiebreak_loss=val_loss)

        if best_improved:
            print(f"  -> New best model saved! (AUC: {val_auc:.4f})")

        if should_stop:
            print(f"  -> Early stopping triggered at epoch {epoch}.")
            break

    return save_path

def main():
    opt = parse_option()
    set_seed(opt.seed)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    # Dataloader
    train_loader, val_loader, test_loader = dataset_loader(opt)

    # Model / Losses
    model, criterion, md_loss, otra_loss = set_model_and_losses(opt, device)

    # Optimizer / Scheduler
    optimizer = set_optimizer(opt, model)
    scheduler = setup_scheduler(opt, optimizer)

    # === Training ===
    print(f"Start Training: {opt.method} on {opt.dataset}")
    best_model_path = train_process(opt, train_loader, val_loader, model, criterion, md_loss, otra_loss, optimizer, scheduler, device)

    # === Testing ===
    print("Start Testing...")
    if os.path.exists(best_model_path):
        if torch.__version__ >= "2.0":
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        else:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model file not found! Using current model state.")

    test_result = test(model, test_loader, device, criterion=criterion, num_classes=opt.num_classes)

    # Print test results
    print(f"\n{'=' * 60}")
    print("  Test Results")
    print(f"  ACC : {test_result.accuracy * 100:.2f}%")
    print(f"  AUC : {test_result.auc_roc:.4f}")
    print(f"  F1  : {test_result.f1:.4f}")
    print(f"  Sens: {test_result.recall:.4f}")
    print(f"  Spec: {test_result.specificity:.4f}")
    print(f"{'=' * 60}\n")

    # Save results (TXT)
    results_dir = f"result/{opt.method}/txt"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/{opt.model_name}_results_summary.txt"

    with open(results_path, "a") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M')}] {opt.dataset}\n")
        f.write(
            f"Config: Seed={opt.seed}, Model={opt.model_name}, N={opt.num_view}, "
            f"BS={opt.batch_size}, LR={opt.lr}, Optim={opt.optim}, "
            f"MD={opt.md_loss}(tau={opt.temp}, lam={opt.lambda_param}), "
            f"otra={opt.otra_loss}(gamma={opt.otra_weight}, mode={opt.otra_weight_mode})\n"
        )
        f.write(f"  ACC : {test_result.accuracy * 100:.2f}%\n")
        f.write(f"  AUC : {test_result.auc_roc:.4f}\n")
        f.write(f"  F1  : {test_result.f1:.4f}\n")
        f.write(f"  Sens: {test_result.recall:.4f}\n")
        f.write(f"  Spec: {test_result.specificity:.4f}\n")
        f.write("-" * 30 + "\n")

    print(f"Results saved to {results_path}")
    
    save_results_csv(opt, test_result)
    print("All processes finished.")

if __name__ == "__main__":
    main()
