"""Train an OppActionHead: BC on real recorded opponent (Dreamer) actions.

Loads a Stage-A LeWM checkpoint (JEPA encoder+projector, frozen), reads
fresh data collected via ``scripts/collect_vs_dreamer.py`` (has real
``obs/opp/action`` labels for the actual opponent LeWM must beat -- not a
JVM built-in AI or scripted policy proxy), and trains a small
``OppActionHead`` (see ``agents/lewm/opp_action_head.py``) via
cross-entropy on ``z_t -> a_opp[t]``.

Usage::

    uv run python scripts/train_opp_action_head.py \\
        --ckpt data/lewm_heads_checkpoint_stride5_m4_v3.pt \\
        --data /media/jeovach/Hoctap/leword-opponent \\
        --out data/opp_action_head.pt \\
        --steps 3000

The output checkpoint stores just the head's state_dict + a few metadata
fields; it is NOT merged into the main LeWM checkpoint (keeps the base
checkpoint untouched, per project's "must keep the LeWM checkpoint"
constraint) and is loaded separately at inference time by whatever overlay
consumes it (e.g. an updated ``online_opponent_model``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.opp_action_head import OppActionHead
from leworldgaming.agents.lewm.projector import Projector
from leworldgaming.data.replay_buffer import DataReader
from leworldgaming.utils.image import normalize_imagenet_pixels
from leworldgaming.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="data/lewm_heads_checkpoint_stride5_m4_v3.pt")
    p.add_argument("--data", required=True, help="file or directory of .h5 replay files")
    p.add_argument("--out", default="data/opp_action_head.pt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-actions", type=int, default=56)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    latent_dim = int(cfg.get("latent_dim", 192))

    encoder = Encoder(
        latent_dim=latent_dim,
        image_size=int(cfg.get("encoder_image_size", 224)),
        patch_size=int(cfg.get("encoder_patch_size", 14)),
        embed_dim=int(cfg.get("encoder_embed_dim", 192)),
        depth=int(cfg.get("encoder_depth", 12)),
        num_heads=int(cfg.get("encoder_heads", 3)),
        mlp_ratio=float(cfg.get("encoder_mlp_ratio", 4.0)),
        dropout=float(cfg.get("encoder_dropout", 0.0)),
    ).to(device)
    projector = Projector(
        latent_dim=latent_dim, hidden_dim=int(cfg.get("projector_hidden", 2048))
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    encoder.eval()
    projector.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in projector.parameters():
        p.requires_grad_(False)

    head = OppActionHead(
        latent_dim=latent_dim, hidden_dim=args.hidden_dim, num_actions=args.num_actions
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    reader = DataReader(args.data)
    print(
        f"[train_opp_action_head] data: {reader.num_files} file(s), "
        f"{reader.total_frames} frames, {reader.total_episodes} episodes"
    )
    if not reader.has_pixels():
        raise RuntimeError(
            "OppActionHead training requires pixel data (frozen encoder operates "
            "on pixels) -- re-collect with --pixels."
        )
    if not reader.has_key("obs/opp/action"):
        raise RuntimeError(
            f"{args.data} lacks 'obs/opp/action' -- this data predates the "
            "opponent-action schema field or was collected via a path that "
            "doesn't populate it. Re-collect via scripts/collect_vs_dreamer.py."
        )

    seq_len = 1  # single-frame windows: we only need z_t -> a_opp[t], no rollout.
    starts = reader.valid_seq_starts(seq_len=seq_len, stride=1)
    train_starts, val_starts = starts.split_by_episode(args.val_fraction, args.seed)
    print(f"[train_opp_action_head] train windows: {train_starts.size}, val windows: {val_starts.size}")

    rng = np.random.default_rng(args.seed)

    @torch.no_grad()
    def _encode(pixels_np: np.ndarray) -> torch.Tensor:
        # pixels_np: (B, 1, C, H, W) uint8 -> (B, D)
        b = pixels_np.shape[0]
        flat = pixels_np.reshape(b, *pixels_np.shape[2:])
        pixels = normalize_imagenet_pixels(torch.from_numpy(flat).to(device))
        return projector(encoder(pixels))

    def _batch_loss(starts_subset, batch_size: int) -> tuple[torch.Tensor, float]:
        batch = reader.sample_window(starts_subset, batch_size, seq_len, rng)
        z = _encode(batch["pixels"])  # (B, D)
        target = torch.from_numpy(batch["opp/action"][:, 0].astype(np.int64)).to(device)
        target = target.clamp(0, args.num_actions - 1)
        logits = head(z)
        loss = F.cross_entropy(logits, target)
        acc = float((logits.argmax(dim=-1) == target).float().mean().item())
        return loss, acc

    t0 = time.time()
    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        head.train()
        loss, acc = _batch_loss(train_starts, args.batch_size)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        if step % 50 == 0 or step == 1:
            print(f"step {step:5d}  train_loss={loss.item():.4f}  train_acc={acc:.3f}  ({time.time()-t0:.0f}s)")

        if step % args.val_every == 0 or step == args.steps:
            head.eval()
            with torch.no_grad():
                val_loss, val_acc = _batch_loss(val_starts, min(512, val_starts.size))
            val_loss_f = float(val_loss.item())
            print(f"  [val] step {step:5d}  val_loss={val_loss_f:.4f}  val_acc={val_acc:.3f}")
            history.append({"step": step, "val_loss": val_loss_f, "val_acc": val_acc})
            if val_loss_f < best_val_loss:
                best_val_loss = val_loss_f
                best_state = {k: v.clone() for k, v in head.state_dict().items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "opp_action_head": best_state,
            "hidden_dim": args.hidden_dim,
            "latent_dim": latent_dim,
            "num_actions": args.num_actions,
            "base_ckpt": args.ckpt,
            "data_path": args.data,
            "best_val_loss": best_val_loss,
            "history": history,
        },
        args.out,
    )
    print(f"[train_opp_action_head] saved best (val_loss={best_val_loss:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
