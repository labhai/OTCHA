#!/usr/bin/env bash
# Shared training settings reported in the OTCHA paper.
set -euo pipefail
OTCHA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$OTCHA_ROOT"

COMMON_ARGS=(
    --device "${DEVICE:-0}" --seed "${SEED:-42}"
    --model_name vit_small_r26_s32_224 --pretrained_weights true
    --num_latents 49
    --ot_pos_weight 0.2 --ot_pos_sigma 1.0
    --sinkhorn_iters 20 --sinkhorn_epsilon 0.2 --bin_score_init 0.0
    --normalize_qk true --refine_dropout 0.0
    --return_transport_matrix false
    --md_loss true --temp 4.0 --lambda_param 0.1
    --otra_loss true --otra_weight 0.3
    --image_size 224 --use_clahe false
    --batch_size 32 --num_workers 8 --epochs 100 --patience 20
    --grad_clip_norm 80.0 --optim SGD
    --lr 1e-3 --scheduler cosine --lr_min 1e-6
    --weight_decay 5e-4 --momentum 0.9
)
