"""
VinDr_eval_missing_view.py

VinDr-Mammo 4-view Study-level Missing-view Robustness Evaluation
==================================================================
평가 단위: study-level (환자 단위, label 고정)
평가 목적: "원래 study label(환자/검사 단위의 진짜 label)을,
            view가 일부 누락된 상황에서도 얼마나 잘 맞추나?"

평가 조합 (총 9가지, 모두 L/R 양쪽 유방 커버):
    4-view : L-CC + L-MLO + R-CC + R-MLO  (baseline)
    3-view : drop L-CC / drop L-MLO / drop R-CC / drop R-MLO  (×4)
    2-view : CC-only (L-CC + R-CC) / MLO-only (L-MLO + R-MLO)  (×2)

    Left-only / Right-only / 1-view 조합은 제외.
    → 한쪽 유방만 커버하면 study-level label과 task 정의가 달라짐.
    → 해당 조합은 별도 breast-level 실험(mode='2view' 체크포인트)으로 평가.

Missing view 처리:
    Image-level zero imputation.
    VinDrDataset은 normalize(mean=0.5, std=0.5) 적용 후 반환하므로:
        impute_value=0.0  →  normalized-space 평균 (raw pixel 0.5, 중간 회색) [권장]
        impute_value=-1.0 →  normalized-space min  (raw pixel 0.0, 완전 검정)

View Index Convention (VinDrDataset._getitem_4view 기준):
    0: L-CC  1: L-MLO  2: R-CC  3: R-MLO

Usage:
    python VinDr_eval_missing_view.py \\
        --ckpt_paths result/.../seed0.pth result/.../seed42.pth ... \\
        --method OTCHA --device 0
"""

from __future__ import annotations

import os
import csv
import argparse
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset.vindr_mammo import VinDrDataset
from models.OTCHA.otcha import OTCHA
from utils import set_seed, str2bool


# =============================================================================
# View combination definitions
# =============================================================================
VIEW_NAMES = {0: "L-CC", 1: "L-MLO", 2: "R-CC", 3: "R-MLO"}


def build_view_combinations() -> List[Dict]:
    """
    Study-level 평가에 유효한 view 조합만 정의.
    조건: L/R 양쪽 유방을 모두 커버해야 study-level label과 task가 일치함.

    Returns:
        List of {"name": str, "active": List[int], "n_views": int, "group": str}
    """
    combos = []

    # ── 4-view (baseline) ────────────────────────────────────────────────────
    combos.append({
        "name":    "4-view (full)",
        "active":  [0, 1, 2, 3],
        "n_views": 4,
        "group":   "4-view",
    })

    # ── 3-view: 한 view씩 제거 (4가지) ────────────────────────────────────────
    # 3-view는 어떤 view를 빼도 반드시 L/R 양쪽 유방이 남음
    for drop_idx in range(4):
        active = [v for v in range(4) if v != drop_idx]
        combos.append({
            "name":    f"3-view (drop {VIEW_NAMES[drop_idx]})",
            "active":  active,
            "n_views": 3,
            "group":   "3-view",
        })

    # ── 2-view: L/R 양쪽 유방을 커버하는 조합만 ─────────────────────────────
    # CC-only  : L-CC(0) + R-CC(2)  → L/R 모두 커버 ✅
    # MLO-only : L-MLO(1) + R-MLO(3) → L/R 모두 커버 ✅
    # Left-only / Right-only는 한쪽 유방만 → 제외
    combos.append({
        "name":    "2-view (CC-only)",
        "active":  [0, 2],
        "n_views": 2,
        "group":   "2-view",
    })
    combos.append({
        "name":    "2-view (MLO-only)",
        "active":  [1, 3],
        "n_views": 2,
        "group":   "2-view",
    })

    return combos


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate_one_combo(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    active_views: List[int],
    impute_value: float = 0.0,
) -> Dict:
    """
    active_views에 없는 view를 impute_value로 채운 뒤 AUROC 계산.
    label은 DataLoader가 반환하는 study-level label 그대로 사용.

    Returns:
        {"auroc": float, "n_pos": int, "n_total": int}
    """
    model.eval()

    missing_views = sorted(set(range(4)) - set(active_views))

    all_probs:  List[float] = []
    all_labels: List[int]   = []

    for batch in test_loader:
        images, targets, _ = batch
        images  = images.to(device, non_blocking=True)   # [B, 4, C, H, W]
        targets = targets.to(device, non_blocking=True)  # [B]

        # ── Image-level zero imputation ──────────────────────────────────────
        # clone()으로 원본 배치 보존 (다음 combo 평가에 영향 없음)
        if missing_views:
            images = images.clone()
            images[:, missing_views, :, :, :] = impute_value

        # ── Forward pass ─────────────────────────────────────────────────────
        # OTCHA.forward() 반환 구조:
        #   output_dict["single"]["logits"]        : single-view branch
        #   output_dict["mv_collection"]["logits"] : OTCHA cross-view branch
        # missing-view 실험 핵심 논거(OT dustbin이 관여하는 branch) → mv_collection
        outputs = model(images)
        logits  = outputs["mv_collection"]["logits"]   # [B, num_classes]

        probs = torch.softmax(logits.float(), dim=-1)[:, 1]  # class-1 probability

        valid_mask = targets != -1
        if valid_mask.sum() == 0:
            continue

        all_probs.extend(probs[valid_mask].cpu().numpy().tolist())
        all_labels.extend(targets[valid_mask].cpu().numpy().tolist())

    n_total = len(all_labels)
    n_pos   = int(sum(all_labels))

    if len(set(all_labels)) < 2:
        return {"auroc": float("nan"), "n_pos": n_pos, "n_total": n_total}

    return {
        "auroc":   float(roc_auc_score(all_labels, all_probs)),
        "n_pos":   n_pos,
        "n_total": n_total,
    }


def evaluate_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    combos: List[Dict],
    impute_value: float = 0.0,
) -> Dict[str, Dict]:
    """
    하나의 체크포인트에 대해 모든 view 조합을 평가.
    Returns: {combo_name: {"auroc": float, "n_pos": int, "n_total": int}}
    """
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
            combo["active"], impute_value,
        )
        results[combo["name"]] = result
        active_str = "+".join(VIEW_NAMES[v] for v in combo["active"])
        print(
            f"    [{combo['name']:30s}] active=[{active_str:23s}] "
            f"AUROC={result['auroc']:.4f}  "
            f"({result['n_pos']}/{result['n_total']} pos)"
        )

    return results


# =============================================================================
# Aggregation & Reporting
# =============================================================================
def aggregate_results(
    all_seed_results: List[Dict[str, Dict]],
    combos: List[Dict],
) -> Dict[str, Dict]:
    """
    N-seed 결과를 mean ± std로 집계.
    Δ = Full 4-view AUROC - current combo AUROC (per seed, then average).
    Returns: {combo_name: {"mean", "std", "delta_mean", "delta_std", "n_pos", "n_total", ...}}
    """
    full_name = "4-view (full)"

    per_combo_auroc: Dict[str, List[float]] = {c["name"]: [] for c in combos}
    per_combo_meta:  Dict[str, Dict]        = {}

    for seed_result in all_seed_results:
        for name, res in seed_result.items():
            if name in per_combo_auroc:
                per_combo_auroc[name].append(res["auroc"])
                per_combo_meta[name] = {"n_pos": res["n_pos"], "n_total": res["n_total"]}

    full_aucs = np.array(per_combo_auroc[full_name])

    aggregated: Dict[str, Dict] = {}
    for combo in combos:
        name  = combo["name"]
        aucs  = np.array(per_combo_auroc[name])
        valid = ~np.isnan(aucs)
        av    = aucs[valid]
        fv    = full_aucs[valid]
        meta  = per_combo_meta.get(name, {"n_pos": 0, "n_total": 0})

        aggregated[name] = {
            "mean":       float(np.mean(av))               if len(av) > 0 else float("nan"),
            "std":        float(np.std(av,  ddof=1))       if len(av) > 1 else 0.0,
            "delta_mean": float(np.mean(fv - av))          if len(av) > 0 else float("nan"),
            "delta_std":  float(np.std(fv - av, ddof=1))   if len(av) > 1 else 0.0,
            "n_pos":      meta["n_pos"],
            "n_total":    meta["n_total"],
            "n_views":    combo["n_views"],
            "group":      combo["group"],
        }

    return aggregated


def print_results_table(method_name: str, aggregated: Dict[str, Dict], combos: List[Dict]):
    print(f"\n{'=' * 90}")
    print(f"  {method_name} | VinDr-Mammo 4-view | Study-level Missing-view Robustness")
    print(f"  Δ = Full 4-view AUROC − combo AUROC  (↑ = 더 많이 하락)")
    print(f"  n_pos/n_total: study-level label 기준 (모든 조합 동일)")
    print(f"{'=' * 90}")
    print(f"  {'Combination':<30} {'#V':>2}  {'AUROC (mean±std)':>18}  "
          f"{'Δ (mean±std)':>18}  {'n_pos/n_total':>14}  Group")
    print(f"  {'-' * 85}")

    for combo in combos:
        name = combo["name"]
        agg  = aggregated[name]

        if not np.isnan(agg["mean"]):
            auc_str   = f"{agg['mean']:.4f} ± {agg['std']:.4f}"
            delta_str = f"{agg['delta_mean']:.4f} ± {agg['delta_std']:.4f}"
        else:
            auc_str   = "      N/A       "
            delta_str = "      N/A       "

        pos_str = f"{agg['n_pos']}/{agg['n_total']}"
        print(
            f"  {name:<30} {agg['n_views']:>2}  "
            f"{auc_str:>18}  {delta_str:>18}  {pos_str:>14}  {agg['group']}"
        )

    print(f"{'=' * 90}\n")


def save_results_csv(
    method_name: str,
    aggregated: Dict[str, Dict],
    combos: List[Dict],
    save_dir: str,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{method_name}_missing_view_studylevel.csv")

    fieldnames = [
        "combination", "n_views", "group",
        "auroc_mean", "auroc_std",
        "delta_mean", "delta_std",
        "n_pos", "n_total", "prevalence",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for combo in combos:
            name = combo["name"]
            agg  = aggregated[name]
            prev = agg["n_pos"] / agg["n_total"] if agg["n_total"] > 0 else float("nan")
            writer.writerow({
                "combination": name,
                "n_views":     agg["n_views"],
                "group":       agg["group"],
                "auroc_mean":  f"{agg['mean']:.6f}",
                "auroc_std":   f"{agg['std']:.6f}",
                "delta_mean":  f"{agg['delta_mean']:.6f}",
                "delta_std":   f"{agg['delta_std']:.6f}",
                "n_pos":       agg["n_pos"],
                "n_total":     agg["n_total"],
                "prevalence":  f"{prev:.4f}",
            })

    print(f"  Results saved to: {csv_path}")
    return csv_path


# =============================================================================
# Argument parsing
# =============================================================================
def parse_option():
    parser = argparse.ArgumentParser(
        description="VinDr-Mammo 4-view Study-level Missing-view Robustness Evaluation"
    )

    parser.add_argument("--ckpt_paths", nargs="+", required=True,
                        help="체크포인트 .pth 경로 목록 (seed 순서대로)")
    parser.add_argument("--method",  type=str, default="OTCHA",
                        help="Method name (출력/파일명 구분용)")
    parser.add_argument("--device",  type=int, default=0)

    # Data
    parser.add_argument("--data_dir",    type=str, default="data/VinDr_Mammo")
    parser.add_argument("--csv_file",    type=str, default="breast-level_annotations.csv")
    parser.add_argument("--splits_json", type=str, default="splits.json")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--use_clahe",   type=str2bool, default=False)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    # Model (학습 시와 반드시 동일하게)
    parser.add_argument("--model_name",         type=str,      default="vit_small_r26_s32_224")
    parser.add_argument("--num_classes",         type=int,      default=2)
    parser.add_argument("--num_latents",         type=int,      default=64)
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
    parser.add_argument("--save_dir", type=str, default="result/missing_view")
    parser.add_argument(
        "--impute_value", type=float, default=0.0,
        help=(
            "missing view slot에 채울 값. "
            "Dataset은 normalize(mean=0.5, std=0.5) 적용 상태이므로: "
            "0.0 = normalized-space 평균 (raw pixel 0.5) [권장], "
            "-1.0 = normalized-space min (raw pixel 0.0)"
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    opt = parse_option()
    set_seed(42)  # 평가 재현성 고정 (학습 seed와 무관)

    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"  VinDr-Mammo Study-level Missing-view Evaluation")
    print(f"  Method : {opt.method}")
    print(f"  Device : {device}")
    print(f"  Seeds  : {len(opt.ckpt_paths)}")
    print(f"  Impute : {opt.impute_value} "
          f"(raw pixel ≈ {opt.impute_value * 0.5 + 0.5:.2f})")
    print(f"{'=' * 60}")

    # ── Test DataLoader ───────────────────────────────────────────────────────
    test_dataset = VinDrDataset(
        data_dir=opt.data_dir,
        csv_file=opt.csv_file,
        splits_json=opt.splits_json,
        split="test",
        mode="4view",
        image_size=opt.image_size,
        expand_channels=3,
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

    # ── Model ─────────────────────────────────────────────────────────────────
    # pretrained_weights=False: 체크포인트에서 완전히 덮어쓰므로 불필요
    model = OTCHA(
        opt=opt,
        model_name=opt.model_name,
        num_classes=opt.num_classes,
        n=4,
        pretrained_weights=False,
        sinkhorn_iters=opt.sinkhorn_iters,
        sinkhorn_epsilon=opt.sinkhorn_epsilon,
        bin_score_init=opt.bin_score_init,
        normalize_qk=opt.normalize_qk,
        refine_dropout=opt.refine_dropout,
        detach_transport=opt.detach_transport,
        ot_pos_weight=opt.ot_pos_weight,
        ot_pos_sigma=opt.ot_pos_sigma,
        return_otcha_info=False,        # 평가 시 info 불필요 → 메모리 절약
        return_transport_matrix=False,
        num_otcha_stages=opt.num_otcha_stages,
    ).to(device)

    # ── View combinations (study-level 유효 조합만) ───────────────────────────
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
            impute_value=opt.impute_value,
        )
        all_seed_results.append(seed_result)

    # ── Aggregation & Report ──────────────────────────────────────────────────
    aggregated = aggregate_results(all_seed_results, combos)
    print_results_table(opt.method, aggregated, combos)
    save_results_csv(opt.method, aggregated, combos, opt.save_dir)

    print("Done.")


if __name__ == "__main__":
    main()