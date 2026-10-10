"""Check the released model/loss API using synthetic tensors on CPU."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss.chexpert_md_loss import MutualDistillationLoss as CheXpertMDLoss
from loss.otra_loss import MutualDistillationLoss, OTRALoss
from models.OTCHA.otcha import OTCHA


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    opt = SimpleNamespace(image_size=32, num_latents=4, latent_refine_iters=1)
    cases = [(2, 2, 0), (4, 2, 0), (2, 2, 1), (2, 13, 0)]
    for views, classes, stages in cases:
        model = OTCHA(
            opt, model_name="vit_tiny_patch16_224", num_classes=classes, n=views,
            pretrained_weights=False, sinkhorn_iters=5, sinkhorn_epsilon=0.2,
            ot_pos_weight=0.2, num_otcha_stages=stages,
            return_otcha_info=True, return_transport_matrix=True,
        )
        inputs = torch.randn(2, views, 3, 32, 32)
        output = model(inputs)
        multi = output["mv_collection"]["logits"]
        single = output["single"]["logits"].reshape(2, views, classes)
        assert multi.shape == (2, classes)
        assert len(output["otcha_stages"]) == stages + 1
        if classes == 13:
            targets = torch.randint(0, 2, (2, classes)).float()
            loss = F.binary_cross_entropy_with_logits(multi, targets)
            loss = loss + CheXpertMDLoss()(multi, single, targets)
        else:
            targets = torch.tensor([0, 1])
            loss = F.cross_entropy(multi, targets)
            loss = loss + MutualDistillationLoss()(multi, single, targets)
        loss = loss + 0.3 * OTRALoss()(output)
        assert torch.isfinite(loss).item(), "Non-finite loss"
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads)
        assert model.latent_tokens.grad is not None
        print(f"PASS: views={views}, classes={classes}, mid_stages={stages}, loss={loss.item():.4f}")
    print("Synthetic model/loss checks passed (CPU, no data or weight downloads).")


if __name__ == "__main__":
    main()
