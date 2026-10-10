#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "${PYTHON:-python}" MURA_otcha.py "${COMMON_ARGS[@]}" \
    --dataset mura --num_view 2 --num_classes 2 \
    --data_dir "${DATA_DIR:-data/MURA-v1.1}" \
    --train_csv mura_train.csv --valid_csv mura_valid.csv --test_csv mura_test.csv \
    --sinkhorn_epsilon 0.1 --lambda_param 0.1 --lr 1e-3 "$@"
