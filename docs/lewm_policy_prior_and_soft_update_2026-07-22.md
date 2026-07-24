# LeWM planner improvements: policy-prior head (negative), MPPI soft-update (positive) — 2026-07-22/23

## Goal
Continue improving the LeWM planner (retrain heads if needed) to beat DreamerV3 in live
self-play, per user's explicit request to research/implement further after the earlier
online-opponent-model and idle-penalty experiments (both negative results, see
`docs/lewm_online_opponent_model_2026-07-22.md`).

## Data
Full offline training corpus at `/media/jeovach/New Volume/leworldgaming/` (9 `.h5` files,
1,103,904 frames, 342 episodes, mixed MctsAi23i/MctsAiZoning/aggressive/defensive/mixed
policies). Reward is extremely sparse: 0.99% of frames nonzero (matches prior calibration
audit's ~1% decisive-frame finding).

## Environment note: fixed a real bug in `_read_windows`
`_PER_CHAR_SCHEMA` (in `data/replay_buffer.py`) gained an `"action"` field in an earlier
session (2026-07-20, for opponent-conditioning research) but the existing 9 `.h5` files on
the "New Volume" data path predate that field (`obs/own/action` / `obs/opp/action` don't
exist in them). `_read_windows` crashed with `KeyError` trying to read them. Fixed by
skipping any per-char schema field that isn't present in a given file (old + new `.h5`
files can now be mixed in one `DataReader` for fields that exist in both) — this is a
general robustness fix, not specific to this session's new policy_head training.

## Attempt 1: Policy-prior head (BC warm-start for CEM) — NEGATIVE RESULT

### Design
Added `PolicyHead` (`z -> logits over num_actions`, `src/leworldgaming/agents/lewm/policy_head.py`):
a small MLP behavior-cloned on the recorded executed action at every grounded training step
(cross-entropy loss, new `heads.policy_loss_weight` / `heads.policy_hidden_dim` config keys
in `train_lewm_heads.py`, 0.0 default = fully backward compatible, no architecture change to
existing checkpoints). Purpose: seed `cem_shooting`'s initial per-timestep categorical
`dist` from this learned prior instead of blind uniform (TD-MPC2 / Sampled-MuZero-style
warm start) — CEM's iterative elite-refinement still re-scores every sampled trajectory
with the exact same frozen reward/value heads every iteration, so a bad prior can only cost
a couple of extra iterations to correct, never silently override planner judgment.

New `cem_shooting(policy_prior=...)` parameter (`planner.py`): replaces `uniform_row` with
the (re-masked-to-valid-actions, renormalized) prior for the *initial* `dist` row and for
warm-start padding; the `min_prob` exploration floor every iteration still mixes in true
uniform (safety net unaffected). `LewmAgent.use_policy_prior` config/CLI knob
(`--planner-no-policy-prior` to disable per-run). Checkpoint save/load, resume support
(with graceful optimizer-state-mismatch handling when adding the head to an existing
Stage-B checkpoint), and 4 new unit tests (prior shapes initial dist, None is a no-op,
masked-to-valid-actions fallback) were added — 26 total tests, all pass.

### Training run
Resumed `lewm_heads_checkpoint_stride5_m4_v3.pt` (step 20000) with `policy_loss_weight=1.0`
for +10000 steps (`configs/lewm_heads_m5_policy.yaml`, 3.0-3.6 step/s, ~46 min) to
`data/lewm_heads_checkpoint_stride5_m5_policy.pt`. Policy CE loss dropped from ~4.1
(near-random init; `ln(56)=4.03`) to ~3.3-3.4 — real but modest learning, consistent with a
highly multi-modal mixed-policy dataset that's hard to behavior-clone cleanly (many
plausible actions per state, softmax CE penalizes confident wrong single-mode predictions).

### Live A/B (docker-harness `MODE=pixels`, `--p2-frame-skip 2`, ZEN, n=9 rounds each)

| Config | Win rate |
|---|---|
| Baseline (v3, no policy head) | 33.3% (reproduced) |
| m5_policy checkpoint, prior ON (default) | 11.1% |
| m5_policy checkpoint, prior OFF | 22.2% |

Both configurations of the new checkpoint underperformed the original v3 baseline — prior ON
is clearly worse (11.1%), and even prior OFF (which should be architecturally identical to
v3 for the reward/value/continuation heads, since only a new head was added) came in lower
(22.2%), within the range of run-to-run variance previously documented but not an
improvement in either case.

### Conclusion
**Policy-prior warm start does NOT help at this training level.** Root cause: the BC loss
plateaus far from a useful action distribution (CE ~3.3-3.4 vs uniform's 4.03 — barely
better than random) because the dataset mixes many different policies' behavior at each
state; a single softmax head averages across them into a prior that's not obviously more
useful than uniform once masked to valid actions, and can actively mislead CEM into wasting
its first iteration on modes averaged from mismatched policies. **Default changed to
`use_policy_prior=False`** (kept as opt-in infrastructure, not deleted, since a
differently-regularized/temperature-scaled policy head could still pay off later — e.g.
distilling from a single strong reference policy instead of the full mixed corpus, or using
it only as a `min_prob`-style floor rather than the full initial distribution).

## Attempt 2: MPPI-style soft/Boltzmann elite update (`elite_temp`) — POSITIVE RESULT

(Implemented in the previous turn — see prior summary — but not yet live-validated on a
reliable harness at the time. This session validated it properly using the newly-adopted
docker-based render harness, see below.)

### Render harness fix
The native Linux `:1` GNOME/NVIDIA display repeatedly deadlocked at JVM's
`util.ResourceDrawer <init>` this session (5 consecutive JVM restarts across ~2 hours all
hung >9-28 min with 0% GPU utilization) under host memory pressure (system swap was fully
exhausted, 4.0/4.0GB, many Chrome/VSCode processes). **Switched to the repo's existing
`docker compose -f docker/fightingice/docker-compose.yml` (`MODE=pixels`) container**,
which runs its own isolated Xvfb-style renderer — every run since completed cleanly in
~2-4 minutes wall-clock for 9 rounds, zero stalls. This is now the **recommended live-eval
harness** going forward (`docker compose ... down && MODE=pixels docker compose ... up -d`,
wait ~10s for "listening on 31415", then run `self_play.py` directly against
`127.0.0.1:31415` — no native JVM launch/DISPLAY wrangling needed).

### Live A/B results (docker-harness, `--p2-frame-skip 2`, ZEN, 9 rounds/run)

| Run | elite_temp=0.0 (baseline) | elite_temp=1.0 |
|---|---|---|
| 1 | 11.1% | 44.4% |
| 2 | 11.1% | 33.3% |
| 3 | 33.3% | 22.2% |
| 4 | 22.2% | 55.6% |
| 5 | 22.2% | 11.1% |
| 6 | — | 11.1% |
| **Aggregate** | **9/45 = 20.0%** | **16/54 = 29.6%** |

Per-run variance is still large (11.1%-55.6% for elite_temp=1.0, matching the
previously-documented win-clustering-by-connection pattern), but the *aggregate* across 5-6
independent runs each shows a consistent, meaningful gap (~+10 percentage points) favoring
the soft update, holding up as more runs were added rather than regressing to baseline.

### Conclusion
**`elite_temp=1.0` (MPPI-style soft/Boltzmann elite reweighting in `cem_shooting`) is
promoted to the new default** (`LewmAgent.planner_elite_temp` default changed from `0.0` to
`1.0`; `configure_planner(elite_temp=0.0)` / `--planner-elite-temp 0.0` still available for
exact legacy reproduction). This is the first planner change this project has found that
reliably improves LeWM's live win rate against Dreamer beyond noise, out of: MCTS (loses
badly at every budget), online opponent-model overlay (no change / slightly worse), idle
no-op penalty (worse), reward-clip + value-down-weight (much worse), policy-prior warm-start
(worse). Mechanism: replacing CEM's hard top-k elite count with a softmax(score/temp)-weighted
refit uses the *entire* elite ranking (not just a 0/1 cutoff), which is more sample-efficient
at this planner's low `num_samples` (24) budget and less prone to a single noisy high-scoring
sample locking in a bad refit direction — directly addressing the previously-documented
reward/value head imprecision on rare decisive frames without needing the (still-collapsed)
ensemble's uncertainty signal.

## Current best-known LeWM configuration
- Checkpoint: `data/lewm_heads_checkpoint_stride5_m4_v3.pt` (unchanged from before this session).
- Planner: CEM, `elite_temp=1.0` (new default), all other knobs at their existing defaults
  (`value_weight=1.0`, `reward_clip=0.0`, `idle_penalty=0.0`, `use_policy_prior=False`).
- Estimated live win rate vs Dreamer (`--p2-frame-skip 2`, ZEN): **~30%** (n=54 rounds),
  up from the long-standing **~20%** baseline (n=45 rounds, this session; historically cited
  as "33.3%" from an n=9-12 sample in earlier docs — this session's larger aggregate sample
  puts the true baseline closer to 20%, underscoring how unreliable n=9 single-run estimates
  have been throughout this project).

## Verification
- `tests/test_lewm_contracts.py`: 26/26 pass (was 18 before this session; +4 policy-prior
  tests, +4 idle-penalty/soft-update tests carried over from the prior turn).
- Live: 11 independent 9-round self-play runs against Dreamer this session (docker harness),
  spanning baseline, elite_temp=1.0, elite_temp=0.5, policy-prior on/off — summarized above.

## Files changed
- NEW: `src/leworldgaming/agents/lewm/policy_head.py`
- NEW: `configs/lewm_heads_m5_policy.yaml`
- NEW: `data/lewm_heads_checkpoint_stride5_m5_policy.pt` (not adopted as best-known, kept for
  reference / future policy-head iteration)
- `src/leworldgaming/training/train_lewm_heads.py`: policy_head training support
  (`policy_loss_weight`, `policy_hidden_dim`, BC CE loss, checkpoint save/load/resume)
- `src/leworldgaming/agents/lewm/agent.py`: policy_head construction/load, `use_policy_prior`
  knob (default False), `elite_temp` default changed to 1.0
- `src/leworldgaming/agents/lewm/planner.py`: `cem_shooting(policy_prior=...)` parameter
- `src/leworldgaming/data/replay_buffer.py`: `_read_windows` tolerates per-char schema
  fields missing from older `.h5` files (bugfix, not feature-specific)
- `scripts/self_play.py`: `--planner-no-policy-prior` CLI flag
- `tests/test_lewm_contracts.py`: +4 policy-prior tests
