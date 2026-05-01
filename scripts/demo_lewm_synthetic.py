"""Synthetic LeWM smoke demo — runs end-to-end on Mac CPU/MPS.

Trains a tiny encoder + predictor for a handful of steps on random tensors
to verify torch, the package layout, AMP, and the SIGReg term all wire up.
This intentionally does NOT touch DareFightingICE — it's a toolchain check.

Run:
    uv run python scripts/demo_lewm_synthetic.py
"""

from __future__ import annotations

import torch
from torch import nn

from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.sigreg import sigreg_loss
from leworldgaming.utils.device import amp_autocast, best_device
from leworldgaming.utils.seed import set_seed


def main() -> None:
    set_seed(0)
    device = best_device()
    print(f"[demo] device: {device}")

    batch_size = 4
    image_size = 64  # smaller than 224 for speed on CPU/MPS
    action_dim = 56
    latent_dim = 256

    encoder = Encoder(latent_dim=latent_dim).to(device)
    predictor = Predictor(latent_dim=latent_dim, action_dim=action_dim).to(device)
    params = list(encoder.parameters()) + list(predictor.parameters())
    optim = torch.optim.AdamW(params, lr=3e-4)

    n_params = sum(p.numel() for p in params) / 1e6
    print(f"[demo] params: {n_params:.2f}M (encoder + predictor)")

    for step in range(5):
        o_t = torch.randn(batch_size, 3, image_size, image_size, device=device)
        o_tp1 = torch.randn(batch_size, 3, image_size, image_size, device=device)
        a_t = torch.randn(batch_size, action_dim, device=device)

        with amp_autocast(device):
            z_t = encoder(o_t)
            z_tp1 = encoder(o_tp1)
            z_tp1_pred = predictor(z_t, a_t)

            pred_loss = nn.functional.mse_loss(z_tp1_pred, z_tp1.detach())
            reg_loss = sigreg_loss(z_t)
            loss = pred_loss + 0.1 * reg_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        print(
            f"[demo] step={step} loss={loss.item():.4f} "
            f"pred={pred_loss.item():.4f} sigreg={reg_loss.item():.4f}"
        )

    print("[demo] OK")


if __name__ == "__main__":
    main()
