# PETS final model training on the real vs-Dreamer dataset (2026-07-24)

## Goal
Train a final PETS (Probabilistic Ensembles with Trajectory Sampling)
checkpoint on the fresh, real DreamerV3-opponent dataset
(`/media/jeovach/Hoctap/leword-opponent`, 21 `.h5` files, 267,812 transitions,
117 episodes, real opponent-action labels + pixels) — the same dataset used
for the LeWM head-retrain experiment this session — as a comparison point to
LeWM's planner. PETS is the project's other from-scratch trajectory-shooting
planner (state-vector-based ensemble dynamics + discrete CEM, see
`src/leworldgaming/agents/pets/`), previously never trained on real Dreamer
opponent data (only smoke-tested on a 4,910-transition toy `data/replay.h5`).

## Config
New `configs/pets_vsdreamer.yaml`: identical architecture/hyperparameters to
`configs/pets.yaml` (5-member ensemble, hidden=200, 3 layers, CEM planner
horizon=15/candidates=200/elites=20/iters=4), only `data_path` (fresh
dataset) and `ckpt_path` (`data/pets_checkpoint_vsdreamer.pt`) changed.

## Bug fix: added best-val-checkpoint snapshotting
`train_pets.py` previously had no best-val tracking (unlike
`train_lewm_heads.py`) — it only saved the final-step state (plus periodic
`ckpt_every` overwrites of the *same* file, losing all earlier snapshots).
The ensemble dynamics loss on this dataset is genuinely noisy per-batch
(rare heavy-tailed HP-swing transitions spike NLL/MSE far above the running
mean — a normal characteristic of Gaussian-NLL ensemble training, not a bug)
and validation MSE plateaued/mildly worsened after ~step 9500 (best
val_delta_mse=359.73) despite training to 20000 steps (final val_delta_mse=
389.08 at step 20000 — a real, if modest, overfit past the optimum). Added
the same best-val-state-dict-restore-before-final-save pattern already used
for `train_lewm_heads.py`'s continuation head. Verified via a 60-step smoke
test that the correct (non-final) step gets swapped in and logged.

## Training run
20,000 steps, batch_size=256, ~1.5-1.7 step/s, ~3.6 hours total, RTX 5060 Ti.
Final saved checkpoint (`data/pets_checkpoint_vsdreamer.pt`) has its dynamics
ensemble weights swapped to the **step=9500 snapshot** (val_delta_mse=
359.73), not the step=20000 final state.

Validation MSE trajectory: 407.8 (step 0) -> 359.7 (step 9500, best) -> 375-390
(steps 14000-20000, mild overfit plateau). Training loss itself is noisy
(occasional spikes to NLL~100+/MSE~700+ from rare large-HP-delta batches)
but the gradient-clipped (`grad_clip=1.0`) optimizer never diverged.

## Live evaluation vs DreamerV3

### Real-time budget mismatch discovered and corrected
First attempt (`--p1-frame-skip` left at its `pets`-default of 1, i.e. a
fresh decision requested every ~16.7ms) revealed PETS's actual per-decision
latency is **~100.7ms mean** (CEM: horizon=15, 200 candidates, 4 iters, over
a 5-member ensemble) — **100% drop rate** against a 16.7ms budget. This is
an even larger real-time mismatch than LeWM's (46ms mean at stride5). Fixed
by setting `--p1-frame-skip 8` (133.3ms budget), which brought drop_rate down
to 0.1% (i.e. essentially never missing its deadline) — used for the
reported results below. (`--p1-frame-skip 6`, 100ms budget, was tried first
and still had 85-96% drop rate — 8 was the smallest frame_skip that actually
fit PETS's real latency on this hardware.)

### Result: PETS loses every round (0/9), worse than LeWM
Docker harness (`MODE=pixels`), `--p1-frame-skip 8`, `--p2-frame-skip 2`
(Dreamer throttled to the same legacy-comparison protocol used for all
other results in `docs/lewm_opp_action_head_2026-07-23.md`), ZEN, 3 runs x 3
rounds:

| run | rounds | P1 (PETS) wins |
|---|---|---|
| 1 | 3 | 0 |
| 2 | 3 | 0 |
| 3 | 3 | 0 |
| **total** | **9** | **0 (0.0%)** |

All 9 rounds were clean losses (P1 hp=0 at round end every time). This is
worse than LeWM's ~26-40% win rate under the identical protocol (see
`docs/lewm_opp_action_head_2026-07-23.md` Addenda 2-3), and worse than
LeWM's 0/9 at true 60Hz-vs-60Hz is only comparable in outcome, not in
decision cadence — PETS here needed an even coarser frame_skip (8, ~7.5Hz)
than LeWM (5, ~12Hz) just to stop dropping essentially all its decisions.

## Interpretation

1. **PETS's state-vector-only representation is a much weaker world model
   than LeWM's pixel-pretrained JEPA encoder** for this task — no visual
   information, hand-built 26-dim primitive state, much smaller ensemble
   dynamics network (0.5M params) vs LeWM's pretrained encoder/predictor
   stack. This was already the expected outcome going in (PETS exists in
   this repo mainly as a baseline/comparison point, not a competitive
   candidate), and this result is consistent with that framing.
2. **PETS's CEM is also real-time-constrained, and worse than LeWM's**: its
   ~100ms/decision cost is ~2x LeWM's ~46ms, forcing an even coarser
   effective decision rate (frame_skip=8, ~7.5Hz) to stay within budget.
3. Training on the fresh, real Dreamer-opponent dataset did produce a
   healthy, non-diverging dynamics model (val MSE genuinely improved
   407.8->359.7 before the best-val-snapshot logic correctly caught the
   overfit past that point) — the training pipeline itself is sound; the
   live result is a genuine capability gap, not a training bug.

## Conclusion
PETS trained cleanly on the new dataset and the training pipeline gained a
useful robustness fix (best-val checkpointing), but as a planner it is
**not competitive with LeWM** in this live matchup (0% vs LeWM's ~30-40%
under the same protocol), confirming PETS's role in this project as a
baseline comparison rather than a candidate to displace LeWM. No further
PETS-specific tuning is recommended given this clear gap; LeWM (with
`elite_temp=1.0` and, provisionally, the vs-Dreamer-retrained heads) remains
the best-known agent.
