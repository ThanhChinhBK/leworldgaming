# LeWM temporal_stride: 5 → 2 retrain decision

## Context

LeWM (`data/lewm_heads_checkpoint_stride5_m4.pt`, stride=5) still loses to
Dreamer live, even after fixing action-space masking, reward-head
calibration (M4 retrain), and CEM latency/horizon tuning. Root-caused to a
**decision-rate gap**: at `temporal_stride=5` LeWM only re-decides every 5
raw frames (12Hz) vs Dreamer's native 60Hz — a 5x reaction-speed
disadvantage no amount of inference-time planner tuning can close, since
the world model itself only knows how to jump 5 raw frames per predictor
step (`z_{t+5} = predictor(z_t, block_of_5_actions)` — it has no notion of
predicting `z_{t+1}` from a single action).

An ablation (Dreamer forced to `--frame-skip 5`, i.e. same 12Hz cadence as
LeWM, no retrain) showed **LeWM can outperform Dreamer at equal decision
rate** — evidence LeWM's world-model quality is competitive; it's purely
being out-paced.

## Why not stride=1 (fully match Dreamer's 60Hz)?

Real-time lookahead = `horizon × stride` raw frames. Latency budget =
`stride × 16.67ms`. Going from stride 5→1:
- Budget shrinks **5x** (83ms → 16.67ms/decision).
- To keep today's ~0.67s useful lookahead (needed to see special-move
  startup/active windows — the exact reason this repo moved from stride=1
  to stride=5 in the first place, see `configs/lewm.yaml`'s original
  comment), `horizon` must grow **5x** (8 → 40).
- Planner cost scales roughly `samples × iters × horizon`. A 5x deeper
  horizon costs ~5x more compute, which cancels out the "cheap CEM"
  latency savings we freed up earlier this session. At stride=1/horizon=40,
  fitting under 16.67ms requires `samples` ≈ 2-4 — degenerates CEM's elite
  selection (`elite_frac=0.125` on 2-4 samples ≈ picking 1 of 2/4, no real
  search).
- Alternative (shorten lookahead instead, e.g. horizon=8 → 133ms) reproduces
  the exact "too short to see attack startup" problem stride=5 was
  introduced to fix.

**Conclusion:** stride=1 is a lose-lose trade at this planner's compute
budget — the reaction-speed gain would be undercut by either unusably
shallow search or unusably short lookahead.

## Why stride=2 is the right middle ground

- Budget only tightens **2x** (83ms → ~33ms/decision) — much less severe.
- Horizon only needs to grow **2x** to preserve lookahead (8 → ~10-12 for
  ~330-400ms lookahead, still close to the paper's ~417ms reference).
- This is the one point on the curve where the CEM latency work already
  done this session (24 samples/1 iter/8 horizon costing ~38ms at stride=5)
  meaningfully carries over rather than being erased by the tighter budget:
  ~160 sample-steps of search still fit under the new ~33ms budget at
  stride=2 (e.g. `samples=16, iters=1, horizon=10` ≈ 32ms, ~333ms
  lookahead — recommended starting point, to be re-benchmarked on real
  hardware once the stride=2 checkpoint exists, since exact per-step cost
  may shift slightly).
- Reaction rate goes from 12Hz → 30Hz — still short of Dreamer's 60Hz, but
  a real, meaningful improvement without the stride=1 failure modes above.

## What this requires (no shortcuts)

This is a **Stage-A retrain from scratch**, not a heads-only Stage-B tweak:
`temporal_stride` changes the encoder/action-encoder/predictor's entire
training data framing (block size, action-block width, target latents),
so encoder + projector + action_encoder + predictor + pred_proj all need
to re-converge. Stage B (reward/continuation/value heads) must then be
retrained on top of the new Stage-A checkpoint, same as the stride=5 →
M4 recalibration path earlier this session.

Same replay data (`data/replay.h5`) — no new data collection needed.

## Config change made

`configs/lewm.yaml`: `temporal_stride: 5` → `temporal_stride: 2`, with
`planner.horizon`/`planner.num_samples` doc comments updated to the
stride=2 recommendation above. (Note: the `planner:` section in this YAML
is documentation-only today — `train_lewm.py`'s `DEFAULTS` doesn't include
a `planner` key, so it's never actually saved into the checkpoint's config
or read back by `LewmAgent._build_modules`. Real planner defaults live in
`src/leworldgaming/agents/lewm/agent.py`'s `_build_modules`, which will
need a manual follow-up edit — `horizon`/`num_samples` tuned to the new
stride=2 numbers above — after the retrain, mirroring this session's
stride=5 horizon-tuning work.)

## Retrain commands (Stage A, then Stage B) — NOT run yet

```bash
# Stage A (JEPA pretraining) — from scratch at temporal_stride=2.
# Use a distinct ckpt_path so the existing stride=5 checkpoint is preserved
# for A/B comparison.
uv run python scripts/train.py --agent lewm --stage a --steps 50000 \
  --config configs/lewm.yaml --ckpt-path data/lewm_checkpoint_stride2.pt

# Stage B (reward/continuation/value heads) — on top of the new Stage-A
# checkpoint, using the M4 recipe (imagined_horizon>0, imagined_loss_weight=1.0,
# value_loss_weight=1.0) since that recalibration fix is independent of stride
# and should carry over.
uv run python scripts/train.py --agent lewm --stage b --steps 20000 \
  --config configs/lewm_heads_m4.yaml \
  --ckpt-in data/lewm_checkpoint_stride2.pt \
  --ckpt-path data/lewm_heads_checkpoint_stride2_m4.pt
```

After Stage B completes: update `LewmAgent._build_modules`'s planner
defaults (in `src/leworldgaming/agents/lewm/agent.py`) to the stride=2
numbers (`horizon≈10-12, num_samples≈16-24, num_iters=1`), re-benchmark
latency on real hardware (same synthetic per-decision timing approach used
for the stride=5 tuning this session), then re-test live vs Dreamer and
vs MctsAiZoning.
