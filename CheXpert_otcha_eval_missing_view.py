"""
CheXpert_eval_missing_view.py

CheXpert 2-view Missing-view Robustness Evaluation
===================================================
평가 단위: study-level (frontal + lateral → 한 study)
평가 목적: "2-view(frontal+lateral)로 학습된 모델이
            1-view만 주어졌을 때 얼마나 버티나?"

평가 조합 (총 3가지):
    2-view : frontal + lateral  (baseline)
    1-view : frontal only
    1-view : lateral only

View Index Convention (CheXpertDataset.__getitem__ 기준):
    0: frontal
    1: lateral

Label: multilabel [B, 13], U-Zeros policy
Metric: per-label AUROC + mean AUROC (labels with no positive sample → skip)

Missing view 처리:
    Image-level zero imputation.
    CheXpertDataset은 normalize(mean=0.5, std=0.5) 적용 후 반환하므로:
        impute_value=0.0  →  normalized-space 평균 (raw pixel 0.5) [권장]
        impute_value=-1.0 →  normalized-space min  (raw pixel 0.0)

Usage:
    python CheXpert_eval_missing_view.py \\
        --ckpt_paths result/.../seed0.pth result/.../seed42.pth ... \\
        --method OTCHA --device 0
"""

from __future__ import annotations

import os
import csv
import argparse
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset.chexpert import CheXpertDataset, CHEXPERT_LABELS, NUM_LABELS
from models.OTCHA.otcha import OTCHA
from utils import set_seed, str2bool


# =============================================================================
# View combination definitions
# =============================================================================
VIEW_NAMES = {0: "frontal", 1: "lateral"}


def build_view_combinations() -> List[Dict]:
    return [
        {
            "name":    "2-view (frontal+lateral)",
            "active":  [0, 1],
            "n_views": 2,
            "group":   "2-view",
        },
        {
            "name":    "1-view (frontal only)",
            "active":  [0],
            "n_views": 1,
            "group":   "1-view",
        },
        {
            "name":    "1-view (lateral only)",
            "active":  [1],
            "n_views": 1,
            "group":   "1-view",
        },
    ]


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate_one_combo(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    active_views: List[int],
    label_names: List[str],
    impute_value: float = 0.0,
) -> Dict:
    """
    active_views에 없는 view를 impute_value로 채운 뒤
    per-label AUROC + mean AUROC 계산.

    Returns:
        {
            "mean_auc":   float,
            "per_label":  {label_name: float},   # NaN 포함 가능 (양성 없음)
            "n_total":    int,
        }
    """
    model.eval()

    missing_views = sorted(set(range(2)) - set(active_views))
    n_labels = len(label_names)

    # [N, n_labels] 누적
    all_probs:  List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch in test_loader:
        images, targets, _ = batch
        images  = images.to(device, non_blocking=True)   # [B, 2, C, H, W]
        targets = targets.to(device, non_blocking=True)  # [B, n_labels] float32

        # ── Image-level zero imputation ──────────────────────────────────────
        if missing_views:
            images = images.clone()
            images[:, missing_views, :, :, :] = impute_value

        # ── Forward pass ─────────────────────────────────────────────────────
        # OTCHA.forward() → output_dict["mv_collection"]["logits"]
        # CheXpert: logits shape [B, n_labels], BCEWithLogitsLoss 기준
        outputs = model(images)
        logits  = outputs["mv_collection"]["logits"]   # [B, n_labels]

        probs = torch.sigmoid(logits.float())  # [B, n_labels]

        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_probs_arr   = np.concatenate(all_probs,   axis=0)  # [N, n_labels]
    all_targets_arr = np.concatenate(all_targets, axis=0)  # [N, n_labels]

    n_total = len(all_targets_arr)

    # ── Per-label AUROC ───────────────────────────────────────────────────────
    per_label_auc: Dict[str, float] = {}
    valid_aucs: List[float] = []

    for i, name in enumerate(label_names):
        y_true = all_targets_arr[:, i]
        y_prob = all_probs_arr[:, i]

        # 양성 또는 음성이 하나도 없으면 AUROC 계산 불가 → NaN 처리
        if len(np.unique(y_true)) < 2:
            per_label_auc[name] = float("nan")
        else:
            auc = float(roc_auc_score(y_true, y_prob))
            per_label_auc[name] = auc
            valid_aucs.append(auc)

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    return {
        "mean_auc":  mean_auc,
        "per_label": per_label_auc,
        "n_total":   n_total,
    }


def evaluate_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    combos: List[Dict],
    label_names: List[str],
    impute_value: float = 0.0,
) -> Dict[str, Dict]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if torch.__version__ >= "2.0":
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    else:
        state_dict = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state_dict)
    print(f"  Loaded: {os.path.basename(ckpt_path)}")

    results: Dict[str, Dict] = {}
    for combo in combos:
        result = evaluate_one_combo(
            model, test_loader, device,
            combo["active"], label_names, impute_value,
        )
        results[combo["name"]] = result
        active_str = "+".join(VIEW_NAMES[v] for v in combo["active"])
        print(
            f"    [{combo['name']:30s}] active=[{active_str:18s}] "
            f"mAUC={result['mean_auc']:.4f}  (N={result['n_total']})"
        )

    return results


# =============================================================================
# Aggregation & Reporting
# =============================================================================
def aggregate_results(
    all_seed_results: List[Dict[str, Dict]],
    combos: List[Dict],
    label_names: List[str],
) -> Dict[str, Dict]:
    """
    N-seed 결과를 mean ± std로 집계.
    Δ = 2-view (full) mean_auc - current combo mean_auc (per seed, then average).
    per_label도 seed 평균으로 집계.
    """
    full_name = "2-view (frontal+lateral)"

    # combo별 seed mean_auc 수집
    per_combo_mean_auc: Dict[str, List[float]] = {c["name"]: [] for c in combos}
    # combo별 per-label auc: {combo_name: {label: [auc_seed0, auc_seed1, ...]}}
    per_combo_label_auc: Dict[str, Dict[str, List[float]]] = {
        c["name"]: {lb: [] for lb in label_names} for c in combos
    }
    per_combo_ntotal: Dict[str, int] = {}

    for seed_result in all_seed_results:
        for name, res in seed_result.items():
            if name not in per_combo_mean_auc:
                continue
            per_combo_mean_auc[name].append(res["mean_auc"])
            per_combo_ntotal[name] = res["n_total"]
            for lb in label_names:
                per_combo_label_auc[name][lb].append(res["per_label"].get(lb, float("nan")))

    full_aucs = np.array(per_combo_mean_auc[full_name])

    aggregated: Dict[str, Dict] = {}
    for combo in combos:
        name  = combo["name"]
        aucs  = np.array(per_combo_mean_auc[name])
        valid = ~np.isnan(aucs)
        av    = aucs[valid]
        fv    = full_aucs[valid]

        # per-label mean ± std across seeds
        per_label_agg: Dict[str, Dict] = {}
        for lb in label_names:
            lb_aucs = np.array(per_combo_label_auc[name][lb])
            lb_valid = ~np.isnan(lb_aucs)
            lb_av = lb_aucs[lb_valid]
            per_label_agg[lb] = {
                "mean": float(np.mean(lb_av))         if len(lb_av) > 0 else float("nan"),
                "std":  float(np.std(lb_av, ddof=1))  if len(lb_av) > 1 else 0.0,
            }

        aggregated[name] = {
            "mean":        float(np.mean(av))             if len(av) > 0 else float("nan"),
            "std":         float(np.std(av,  ddof=1))     if len(av) > 1 else 0.0,
            "delta_mean":  float(np.mean(fv - av))        if len(av) > 0 else float("nan"),
            "delta_std":   float(np.std(fv - av, ddof=1)) if len(av) > 1 else 0.0,
            "per_label":   per_label_agg,
            "n_total":     per_combo_ntotal.get(name, 0),
            "n_views":     combo["n_views"],
            "group":       combo["group"],
        }

    return aggregated


def print_results_table(
    method_name: str,
    aggregated: Dict[str, Dict],
    combos: List[Dict],
    label_names: List[str],
):
    print(f"\n{'=' * 85}")
    print(f"  {method_name} | CheXpert | Missing-view Robustness")
    print(f"  Δ = 2-view (full) mean AUROC − combo mean AUROC  (↑ = 더 많이 하락)")
    print(f"{'=' * 85}")
    print(f"  {'Combination':<30} {'#V':>2}  {'mAUROC (mean±std)':>18}  "
          f"{'Δ (mean±std)':>18}  {'N':>6}  Group")
    print(f"  {'-' * 80}")

    for combo in combos:
        name = combo["name"]
        agg  = aggregated[name]

        if not np.isnan(agg["mean"]):
            auc_str   = f"{agg['mean']:.4f} ± {agg['std']:.4f}"
            delta_str = f"{agg['delta_mean']:.4f} ± {agg['delta_std']:.4f}"
        else:
            auc_str   = "      N/A       "
            delta_str = "      N/A       "

        print(
            f"  {name:<30} {agg['n_views']:>2}  "
            f"{auc_str:>18}  {delta_str:>18}  {agg['n_total']:>6}  {agg['group']}"
        )

    # ── Per-label breakdown ───────────────────────────────────────────────────
    print(f"\n  Per-label AUROC (mean ± std across seeds):")
    print(f"  {'Label':<35}", end="")
    for combo in combos:
        print(f"  {combo['name'][:18]:>18}", end="")
    print()
    print(f"  {'-' * (35 + 20 * len(combos))}")

    for lb in label_names:
        print(f"  {lb:<35}", end="")
        for combo in combos:
            agg = aggregated[combo["name"]]
            lb_agg = agg["per_label"][lb]
            if not np.isnan(lb_agg["mean"]):
                s = f"{lb_agg['mean']:.4f}±{lb_agg['std']:.4f}"
            else:
                s = "    N/A    "
            print(f"  {s:>18}", end="")
        print()

    print(f"{'=' * 85}\n")


def save_results_csv(method_name: str, aggregated: Dict[str, Dict], combos: List[Dict], label_names: List[str], save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)

    # ── Summary CSV (mean AUROC per combo) ────────────────────────────────────
    summary_path = os.path.join(save_dir, f"{method_name}_chexpert_missing_view_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "combination", "n_views", "group",
            "mean_auc_mean", "mean_auc_std",
            "delta_mean", "delta_std", "n_total",
        ])
        writer.writeheader()
        for combo in combos:
            name = combo["name"]
            agg  = aggregated[name]
            writer.writerow({
                "combination":   name,
                "n_views":       agg["n_views"],
                "group":         agg["group"],
                "mean_auc_mean": f"{agg['mean']:.6f}",
                "mean_auc_std":  f"{agg['std']:.6f}",
                "delta_mean":    f"{agg['delta_mean']:.6f}",
                "delta_std":     f"{agg['delta_std']:.6f}",
                "n_total":       agg["n_total"],
            })
    print(f"  Summary saved to: {summary_path}")

    # ── Per-label CSV ─────────────────────────────────────────────────────────
    perlabel_path = os.path.join(save_dir, f"{method_name}_chexpert_missing_view_perlabel.csv")
    fieldnames = ["label"] + [
        f"{combo['name']}_{suffix}"
        for combo in combos
        for suffix in ["auc_mean", "auc_std"]
    ]
    with open(perlabel_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lb in label_names:
            row: Dict = {"label": lb}
            for combo in combos:
                name    = combo["name"]
                lb_agg  = aggregated[name]["per_label"][lb]
                row[f"{name}_auc_mean"] = f"{lb_agg['mean']:.6f}"
                row[f"{name}_auc_std"]  = f"{lb_agg['std']:.6f}"
            writer.writerow(row)
    print(f"  Per-label saved to: {perlabel_path}")

    return summary_path


# =============================================================================
# Argument parsing
# =============================================================================
def parse_option():
    parser = argparse.ArgumentParser(description="CheXpert 2-view Missing-view Robustness Evaluation")

    parser.add_argument("--ckpt_paths", nargs="+", required=True,
                        help="체크포인트 .pth 경로 목록 (seed 순서대로)")
    parser.add_argument("--method",  type=str, default="OTCHA")
    parser.add_argument("--device",  type=int, default=0)

    # Data
    parser.add_argument("--data_dir",        type=str,      default="data/CheXpert")
    parser.add_argument("--csv_file",        type=str,      default="chexpert_label.csv")
    parser.add_argument("--splits_json",     type=str,      default="splits.json")
    parser.add_argument("--image_size",      type=int,      default=224)
    parser.add_argument("--expand_channels", type=int,      default=3)
    parser.add_argument("--use_clahe",       type=str2bool, default=False)
    parser.add_argument("--batch_size",      type=int,      default=16)
    parser.add_argument("--num_workers",     type=int,      default=8)

    # Model (학습 시와 반드시 동일하게)
    parser.add_argument("--model_name",         type=str,      default="vit_small_r26_s32_224")
    parser.add_argument("--num_classes",         type=int,      default=NUM_LABELS)
    parser.add_argument("--num_latents",         type=int,      default=49)
    parser.add_argument("--latent_refine_iters", type=int,      default=1)
    parser.add_argument("--ot_pos_weight",       type=float,    default=0.2)
    parser.add_argument("--ot_pos_sigma",        type=float,    default=1.0)
    parser.add_argument("--sinkhorn_iters",      type=int,      default=5)
    parser.add_argument("--sinkhorn_epsilon",    type=float,    default=0.1)
    parser.add_argument("--bin_score_init",      type=float,    default=0.0)
    parser.add_argument("--normalize_qk",        type=str2bool, default=True)
    parser.add_argument("--refine_dropout",      type=float,    default=0.0)
    parser.add_argument("--detach_transport",    type=str2bool, default=False)
    parser.add_argument("--num_otcha_stages",    type=int,      default=0)

    # Output
    parser.add_argument("--save_dir",     type=str,   default="result/missing_view")
    parser.add_argument("--impute_value", type=float, default=0.0,
                        help="0.0=normalized mean (권장), -1.0=normalized min")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    opt = parse_option()
    set_seed(42)

    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"  CheXpert Missing-view Evaluation")
    print(f"  Method  : {opt.method}")
    print(f"  Device  : {device}")
    print(f"  Seeds   : {len(opt.ckpt_paths)}")
    print(f"  Labels  : {NUM_LABELS}")
    print(f"  Impute  : {opt.impute_value} "
          f"(raw pixel ≈ {opt.impute_value * 0.5 + 0.5:.2f})")
    print(f"{'=' * 60}")

    # ── Test DataLoader ───────────────────────────────────────────────────────
    test_dataset = CheXpertDataset(
        data_dir=opt.data_dir,
        csv_file=opt.csv_file,
        splits_json=opt.splits_json,
        split="test",
        image_size=opt.image_size,
        expand_channels=opt.expand_channels,
        use_clahe=opt.use_clahe,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    # ── Model (n=2 고정) ──────────────────────────────────────────────────────
    model = OTCHA(
        opt=opt,
        model_name=opt.model_name,
        num_classes=opt.num_classes,   # NUM_LABELS (13)
        n=2,                           # frontal + lateral
        pretrained_weights=False,
        sinkhorn_iters=opt.sinkhorn_iters,
        sinkhorn_epsilon=opt.sinkhorn_epsilon,
        bin_score_init=opt.bin_score_init,
        normalize_qk=opt.normalize_qk,
        refine_dropout=opt.refine_dropout,
        detach_transport=opt.detach_transport,
        ot_pos_weight=opt.ot_pos_weight,
        ot_pos_sigma=opt.ot_pos_sigma,
        return_otcha_info=False,
        return_transport_matrix=False,
        num_otcha_stages=opt.num_otcha_stages,
    ).to(device)

    # ── View combinations ─────────────────────────────────────────────────────
    combos = build_view_combinations()
    print(f"\n  View combinations to evaluate: {len(combos)}")
    for c in combos:
        print(f"    {c['name']:30s} active={[VIEW_NAMES[v] for v in c['active']]}")

    # ── Per-seed evaluation ───────────────────────────────────────────────────
    all_seed_results: List[Dict[str, Dict]] = []

    for seed_idx, ckpt_path in enumerate(opt.ckpt_paths):
        print(f"\n[Seed {seed_idx + 1}/{len(opt.ckpt_paths)}]")
        seed_result = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            model=model,
            test_loader=test_loader,
            device=device,
            combos=combos,
            label_names=CHEXPERT_LABELS,
            impute_value=opt.impute_value,
        )
        all_seed_results.append(seed_result)

    # ── Aggregation & Report ──────────────────────────────────────────────────
    aggregated = aggregate_results(all_seed_results, combos, CHEXPERT_LABELS)
    print_results_table(opt.method, aggregated, combos, CHEXPERT_LABELS)
    save_results_csv(opt.method, aggregated, combos, CHEXPERT_LABELS, opt.save_dir)

    print("Done.")


if __name__ == "__main__":
    main()