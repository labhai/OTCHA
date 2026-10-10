#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "${PYTHON:-python}" VinDr_otcha.py "${COMMON_ARGS[@]}" \
    --dataset vindr --num_view 2 --num_classes 2 --data_mode 2view \
    --data_dir "${DATA_DIR:-data/VinDr_Mammo}" \
    --csv_file breast-level_annotations.csv --splits_json splits.json \
    --sinkhorn_epsilon 0.2 --lambda_param 1.0 --lr 1e-3 "$@"
