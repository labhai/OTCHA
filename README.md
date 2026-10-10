# OTCHA: Optimal Transport-driven Confidence-aware Latent Hub Alignment for Multi-View Medical Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Conference: MICCAI 2026](https://img.shields.io/badge/Conference-MICCAI_2026-blue)](https://papers.miccai.org/miccai-2026/paper/4914_paper.pdf)

Official PyTorch implementation of **OTCHA: Optimal Transport-driven Confidence-aware Latent Hub Alignment**.

**Paper:** [OTCHA: Optimal Transport-driven Confidence-aware Latent Hub Alignment for Multi-View Medical Image Classification](https://papers.miccai.org/miccai-2026/paper/4914_paper.pdf)  
**Authors:** [Jiwoong Yang](https://scholar.google.com/citations?user=rrsgcvcAAAAJ&hl=ko), [Haejun Chung](https://scholar.google.com/citations?user=O-oZnIwAAAAJ&hl=ko), and [Ikbeom Jang](https://scholar.google.com/citations?user=1rBh9xkAAAAJ&hl=ko)  
**Conference:** *MICCAI 2026*

<p align="center">
  <img src="./figures/otcha_overview.png" alt="OTCHA Overview" width="85%">
</p>
<p align="center">
  <b>Figure 1.</b> Overview of the proposed OTCHA framework.
</p>

## Abstract

Multi-view imaging, such as mammography and chest radiography, is a standard component of clinical practice. However, medical images are often unregistered and contain view-specific artifacts or irrelevant background cues that can obscure diagnostically relevant findings. Many existing methods directly fuse per-view representations, allowing such irrelevant content to contaminate the fused embedding and reducing robustness under varying view configurations.

We propose **OTCHA**, a confidence-aware latent hub token alignment module based on optimal transport (OT) that refines patch tokens before fusion for multi-view classification. OTCHA introduces a set of learnable latent hub tokens shared across views. For each view, we compute an OT plan between patch tokens and hub tokens that jointly considers feature similarity and geometry, and augment the OT formulation with token-conditional dustbins to enable partial matching and discard irrelevant tokens. The resulting transport plan provides token-wise matching confidence, which gates hub-mediated message passing and weights a novel optimal-transport-based representation alignment loss to stabilize refinement.

Experiments on three multi-view medical image datasets demonstrate consistent improvements over competing baselines across diverse anatomies and view configurations.

## Supported tasks

| Dataset | Views | Task |
| --- | --- | --- |
| VinDr-Mammo | 2 (breast) or 4 (study) | Binary BI-RADS classification |
| MURA | 2 | Musculoskeletal abnormality classification |
| CheXpert | 2 (frontal and lateral) | Classification of 13 observations |

Training and evaluation require no external logging service or account. Metrics
are printed in the terminal; checkpoints and test results are saved locally.

## Installation

Use Python 3.9–3.11. The pinned dependencies were validated with Python 3.9.19,
PyTorch 2.8.0, and timm 1.0.9:

```bash
git clone https://github.com/labhai/OTCHA.git
cd OTCHA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test uses synthetic tensors on CPU. It needs no dataset, GPU, pretrained
weights, or account, and verifies model outputs, losses, and backward passes.
Learning-rate schedulers use PyTorch's built-in components.

For real training, install a PyTorch build compatible with your hardware and
prepare one of the datasets below. Training automatically uses a GPU when
available and falls back to CPU. The default `--pretrained_weights true` downloads
backbone weights through timm on first use; use `--pretrained_weights false` to
initialize without downloading weights.

## Use the model in Python

```python
from types import SimpleNamespace
import torch
from models.OTCHA.otcha import OTCHA

config = SimpleNamespace(image_size=224, num_latents=49, latent_refine_iters=1)
model = OTCHA(config, "vit_small_r26_s32_224", num_classes=2, n=2,
              pretrained_weights=False, return_otcha_info=False)
model.eval()

# Input shape: [batch, views, channels, height, width]
images = torch.randn(1, 2, 3, 224, 224)
with torch.no_grad():
    logits = model(images)["mv_collection"]["logits"]
print(logits.shape)  # torch.Size([1, 2])
```

This example uses random inputs and randomly initialized weights. For meaningful
predictions, use the dataset preprocessing below and load a trained checkpoint.

## Data setup

Images, annotation tables, original split files, and trained checkpoints are
not included. Acquire each dataset separately and prepare the metadata described in
[docs/DATASETS.md](docs/DATASETS.md). All paths are configurable with `--data_dir`
and the corresponding CSV/split options. The expected local layout is:

```text
data/
├── VinDr_Mammo/
│   ├── breast-level_annotations.csv
│   ├── splits.json
│   └── images/<study_id>/<image_id>.png
├── CheXpert/
│   ├── chexpert_label.csv
│   ├── splits.json
│   └── <image paths referenced by the CSV>
└── MURA-v1.1/
    ├── mura_train.csv
    ├── mura_valid.csv
    ├── mura_test.csv
    └── <image paths referenced by the CSV>
```

The files under `examples/` illustrate metadata schemas using fictional IDs;
they are not training data. `data/` is ignored by Git.

## Training

Run from the repository root:

```bash
bash run/vindr_otcha.sh
bash run/chexpert_otcha.sh
bash run/mura_otcha.sh
```

These wrappers use the bundled experiment-script presets and allow CLI overrides
at the end of the command. They run one seed per invocation:

```bash
SEED=43 DEVICE=0 bash run/vindr_otcha.sh --batch_size 16
DATA_DIR=/path/to/CheXpert bash run/chexpert_otcha.sh --num_workers 4
bash run/vindr_otcha.sh --num_view 4 --data_mode 4view
```

| Setting | VinDr-Mammo | CheXpert | MURA |
| --- | --- | --- | --- |
| Views / outputs | 2 / 2 | 2 / 13 | 2 / 2 |
| Latent tokens | 49 | 49 | 49 |
| Sinkhorn iterations / epsilon | 20 / 0.2 | 20 / 0.2 | 20 / 0.1 |
| Mutual distillation weight | 1.0 | 0.1 | 0.1 |
| OTRA weight | 0.3 | 0.3 | 0.3 |
| Learning rate (SGD) | 0.001 | 0.01 | 0.001 |

All presets use `vit_small_r26_s32_224`, image size 224, batch size 32, 100 epochs,
and early stopping patience 20. Direct Python entrypoints have their own defaults;
use the wrappers for the presets above. Available options:

```bash
python VinDr_otcha.py --help
python CheXpert_otcha.py --help
python MURA_otcha.py --help
```

The table above lists the settings shipped in the scripts. The paper's shared
implementation settings specify a learning rate of `1e-3`, mutual distillation
weight `0.1`, Sinkhorn epsilon `0.2`, and cosine annealing. To use those settings:

```bash
bash run/vindr_otcha.sh --lambda_param 0.1 --scheduler cosine
bash run/chexpert_otcha.sh --lr 1e-3 --scheduler cosine
bash run/mura_otcha.sh --sinkhorn_epsilon 0.2 --scheduler cosine
```

Checkpoints are written under `result/OTCHA/<dataset>/save_pth/`. Test summaries
are written under `result/OTCHA/txt/`; VinDr-Mammo and MURA also append CSV results.
Generated results and weights are ignored by Git.

For a short training run over your prepared data, use one epoch:

```bash
bash run/vindr_otcha.sh --epochs 1 --batch_size 2 --num_workers 0
```

## Missing-view evaluation

Use checkpoint paths from your own training runs. Match the backbone, view count,
latent count, stage count, and OT settings to the checkpoint's training configuration.
For checkpoints trained with the provided OT settings:

```bash
python VinDr_otcha_eval_missing_breast.py \
  --ckpt_paths /path/to/vindr_checkpoint.pth \
  --num_latents 49 --sinkhorn_iters 20 --sinkhorn_epsilon 0.2

python VinDr_otcha_eval_missing_study.py \
  --ckpt_paths /path/to/vindr_4view_checkpoint.pth \
  --num_latents 49 --sinkhorn_iters 20 --sinkhorn_epsilon 0.2

python CheXpert_otcha_eval_missing_view.py \
  --ckpt_paths /path/to/chexpert_checkpoint.pth \
  --num_latents 49 --sinkhorn_iters 20 --sinkhorn_epsilon 0.2
```

VinDr breast-level evaluation uses a two-view checkpoint for CC/MLO pairs.
Study-level evaluation uses a four-view checkpoint and one label per study.
Train it with `--num_view 4 --data_mode 4view`. Missing views are imputed with a
constant value. Use `--help` for imputation and output options.

## Structure

```text
models/OTCHA/otcha.py              Core OTCHA model
loss/                             Mutual distillation and OTRA losses
dataset/                          Dataset loaders and image transforms
process/OTCHA/                    Training, validation, testing, and metrics
run/                              Training presets
scripts/smoke_test.py              Synthetic forward/backward check
docs/DATASETS.md                   Metadata formats and preprocessing notes
examples/                         Fictional metadata examples
```

Check model output shapes and finite gradients without data or pretrained downloads:

```bash
python scripts/smoke_test.py
```

Synthetic checks cover model outputs and finite gradients, one-epoch training and
checkpoint reloads for all three datasets, and all three missing-view evaluators.
These checks use fictional images on CPU; they do not establish dataset-level
performance.

## Reproducibility

Exact reproduction requires the original preprocessing, labels, and split
assignments. In particular, the paper uses ROI cropping and background suppression
for VinDr-Mammo, two sampled images per eligible MURA study, and VisualCheXbert
labels for CheXpert. The loaders expect already prepared images and metadata;
see [docs/DATASETS.md](docs/DATASETS.md) for the required formats.

Original split files, dataset images, annotation tables, and trained checkpoints
are not distributed in this repository. The paper reports results over five
training runs; run each seed separately with the `SEED` setting.

## Citation

If you use OTCHA in your research, please cite:

```bibtex
@inproceedings{yang2026otcha,
  title     = {{OTCHA}: Optimal Transport-driven Confidence-aware Latent Hub Alignment for Multi-View Medical Image Classification},
  author    = {Yang, Jiwoong and Chung, Haejun and Jang, Ikbeom},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
  url       = {https://papers.miccai.org/miccai-2026/paper/4914_paper.pdf}
}
```

## License

This project is released under the [MIT License](LICENSE).
