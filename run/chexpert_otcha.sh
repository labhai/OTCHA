#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "${PYTHON:-python}" CheXpert_otcha.py "${COMMON_ARGS[@]}" \
    --dataset chexpert --num_view 2 --num_classes 13 \
    --data_dir "${DATA_DIR:-data/CheXpert}" \
    --csv_file chexpert_label.csv --splits_json splits.json --expand_channels 3 "$@"
