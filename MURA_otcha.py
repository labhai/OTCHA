# main_otcha_mura.py
from __future__ import annotations

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Dataset (MURA)
from dataset.mura import MURADataset

# Network
from models.OTCHA.otcha import OTCHA

# Process
from process.OTCHA.engine import train_one_epoch, validate, test

# Loss
from loss.otra_loss import OTRALoss, MutualDistillationLoss

# Utils
from utils import save_results_csv, EarlyStopper, set_seed, set_optimizer, str2bool, create_scheduler

def parse_option():
    parser = argparse.ArgumentParser(description="OTCHA Training on MURA")

    parser.add_argument("--device", type=int, default=0, help="GPU number")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_view", type=int, default=2, help="number of views")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--dataset", type=str, default="mura", help="dataset name")

    # Method & Model
    parser.add_argument("--method", type=str, default="OTCHA", help="method name")
    parser.add_argument("--model_name", type=str, default="vit_small_r26_s32_224", help="timm model name")
    parser.add_argument("--pretrained_weights", type=str2bool, default=True, help="initialize the backbone with ImageNet weights")

    # fused OT (pos term)
    parser.add_argument("--num_latents", type=int, default=49, help="number of latent hub tokens")
    parser.add_argument("--ot_pos_weight", type=float, default=0.2, help="positive geometry weight for fused OT; must be > 0")
    parser.add_argument("--ot_pos_sigma", type=float, default=1.0, help="sigma for Gaussian-like positional score: -d^2/(2*sigma^2)")

    # OTCHA params
    parser.add_argument("--sinkhorn_iters", type=int, default=20, help="sinkhorn iterations")
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.2, help="sinkhorn epsilon")
    parser.add_argument("--bin_score_init", type=float, default=0.0, help="dustbin logit init")
    parser.add_argument("--normalize_qk", type=str2bool, default=True, help="L2 normalize q/k for cosine scores")
    parser.add_argument("--refine_dropout", type=float, default=0.0, help="dropout on aggregated message")
    parser.add_argument("--return_transport_matrix", type=str2bool, default=False, help="store P in otcha_info (debug)")

    # Loss toggles
    parser.add_argument("--md_loss", type=str2bool, default=True, help="use mutual distillation loss")
    parser.add_argument("--otra_loss", type=str2bool, default=True, help="use the OTRA representation alignment loss")

    # MD hyperparameters
    parser.add_argument("--temp", type=float, default=4.0, help="mutual distillation temperature")
    parser.add_argument("--lambda_param", type=float, default=0.1, help="mutual distillation lambda (inside md_loss)")

    # otra hyperparameters
    parser.add_argument("--otra_weight", type=float, default=0.3, help="gamma: weight for otra loss")

    # Data paths (MURA)
    parser.add_argument("--data_dir", type=str, default="data/MURA-v1.1", help="MURA data directory")
    parser.add_argument("--train_csv", type=str, default="mura_train.csv", help="train CSV file")
    parser.add_argument("--valid_csv", type=str, default="mura_valid.csv", help="valid CSV file")
    parser.add_argument("--test_csv", type=str, default="mura_test.csv", help="test CSV file")
    parser.add_argument("--image_size", type=int, default=224, help="input image size")
    parser.add_argument("--use_clahe", type=str2bool, default=False, help="apply CLAHE")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--grad_clip_norm", type=float, default=80.0, help="grad clip norm")

    # AMP
    parser.add_argument("--use_amp", type=str2bool, default=False, help="use torch.amp autocast + GradScaler")

    # Optimization
    parser.add_argument("--optim", type=str, default="SGD", choices=["Adam", "AdamW", "SGD"], help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"], help="lr scheduler")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="minimum lr")

    opt = parser.parse_args()
    return opt

def dataset_loader(opt):
    train_dataset = MURADataset(
        data_dir=opt.data_dir, csv_file=opt.train_csv,
        split="train", image_size=opt.image_size,
        expand_channels=3, use_clahe=opt.use_clahe,
    )
    val_dataset = MURADataset(
        data_dir=opt.data_dir, csv_file=opt.valid_csv,
        split="val", image_size=opt.image_size,
        expand_channels=3, use_clahe=opt.use_clahe,
    )
    test_dataset = MURADataset(
        data_dir=opt.data_dir, csv_file=opt.test_csv,
        split="test", image_size=opt.image_size,
        expand_channels=3, use_clahe=opt.use_clahe,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers, pin_memory=True,
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
        ot_pos_weight=opt.ot_pos_weight,
        ot_pos_sigma=opt.ot_pos_sigma,
        return_otcha_info=True,
        return_transport_matrix=opt.return_transport_matrix,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    md_loss = None
    if opt.md_loss:
        md_loss = MutualDistillationLoss(temp=opt.temp, lambda_hyperparam=opt.lambda_param).to(device)

    otra_loss = None
    if opt.otra_loss:
        otra_loss = OTRALoss().to(device)

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
        f"eps{opt.sinkhorn_epsilon}_"
        f"M{opt.num_latents}_bin{opt.bin_score_init}_posW{opt.ot_pos_weight}_posS{opt.ot_pos_sigma}_"
        f"md({opt.md_loss})_lam{opt.lambda_param}_"
        f"otra({opt.otra_loss})_gamma{opt.otra_weight}_"
        f"normqk({opt.normalize_qk}).pth"
    )

    early_stopper = EarlyStopper(patience=getattr(opt, "patience", 5), use_loss_when_tie=True, save_path=save_path)

    saved_this_run = False
    for epoch in range(1, opt.epochs + 1):
        start_time = time.time()

        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_classes=opt.num_classes, md_loss=md_loss,
                                       otra_loss=otra_loss, otra_weight=opt.otra_weight, max_grad_norm=opt.grad_clip_norm, use_amp=opt.use_amp)
        val_result = validate(model, val_loader, criterion, device, num_classes=opt.num_classes, epoch=epoch)

        # scheduler step
        if scheduler is not None:
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
            saved_this_run = True
            print(f"  -> New best model saved! (AUC: {val_auc:.4f})")

        if should_stop:
            print(f"  -> Early stopping triggered at epoch {epoch}.")
            break

    if not saved_this_run:
        raise RuntimeError("Training did not produce a checkpoint; check the validation data and metrics")
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
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    print(f"Loaded the best model trained in this run: {best_model_path}")

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
            f"otra={opt.otra_loss}(gamma={opt.otra_weight})\n"
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
