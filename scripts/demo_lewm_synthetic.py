"""Synthetic LeWM smoke demo — runs end-to-end on Mac CPU/MPS.

Trains the full LeWM stack (ViT encoder + projector + AR predictor +
ActionEncoder + SIGReg) for a handful of steps on random tensors to verify
torch, the package layout, AMP, and the loss path all wire up. This
intentionally does NOT touch DareFightingICE — it's a toolchain check.

Run:
    uv run python scripts/demo_lewm_synthetic.py
"""

from __future__ import annotations

import torch
from torch import nn

from leworldgaming.agents.lewm.action_encoder import ActionEncoder
from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.projector import Projector
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
    history_size = 3
    seq_len = history_size + 1

    encoder = Encoder(
        latent_dim=latent_dim, image_size=image_size, patch_size=16, depth=2, num_heads=3
    ).to(device)
    projector = Projector(latent_dim=latent_dim).to(device)
    action_encoder = ActionEncoder(action_dim=action_dim, emb_dim=latent_dim).to(device)
    predictor = Predictor(
        latent_dim=latent_dim,
        action_dim=latent_dim,
        history_size=history_size,
        depth=2,  # small for the demo
    ).to(device)
    pred_proj = Projector(latent_dim=latent_dim).to(device)
    params = (
        list(encoder.parameters())
        + list(projector.parameters())
        + list(action_encoder.parameters())
        + list(predictor.parameters())
        + list(pred_proj.parameters())
    )
    optim = torch.optim.AdamW(params, lr=3e-4)

    n_params = sum(p.numel() for p in params) / 1e6
    print(f"[demo] params: {n_params:.2f}M (encoder+projector+act_enc+predictor+pred_proj)")

    for step in range(5):
        pixels = torch.randn(batch_size, seq_len, 3, image_size, image_size, device=device)
        a_idx = torch.randint(0, action_dim, (batch_size, seq_len), device=device)
        a_oh = nn.functional.one_hot(a_idx, num_classes=action_dim).float()

        with amp_autocast(device):
            b, t = pixels.shape[:2]
            emb = projector(encoder(pixels.reshape(b * t, *pixels.shape[2:]))).reshape(b, t, -1)
            ctx_emb = emb[:, :history_size]
            tgt_emb = emb[:, 1:]
            ctx_act = action_encoder(a_oh[:, :history_size])
            pred = pred_proj(predictor(ctx_emb, ctx_act).reshape(b * history_size, -1)).reshape(
                b, history_size, -1
            )

            pred_loss = nn.functional.mse_loss(pred, tgt_emb.detach())
            reg_loss = sigreg_loss(emb.transpose(0, 1).reshape(t, b, -1))
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
