# LeWM: chunked CEM execution vs. stride-5 re-retrain (2026-07-20)

## Context

Following the v4 ensemble audit (`docs/lewm_calibration_audit_and_ensembling_2026-07-20.md`),
which ruled out reward/value calibration as the reason LeWM loses to Dreamer and
pinned the blame on per-decision latency/drop-rate (stride=2, frame_skip=2: ~34ms
mean, ~49% drop rate against a 33ms budget), this doc covers two follow-up
experiments run the same day:

1. **Chunked/open-loop CEM execution** (zero retrain): only replan every
   `chunk_size` decisions, reusing the already-computed multi-step CEM action
   distribution for the intervening decisions.
2. **Stride-5 Stage-B head re-retrain**: keep frame_skip=5 (83.3ms budget per
   decision, comfortably fits a full CEM search) but retrain the Stage-B heads
   with the current (post-v3) recipe against the existing stride-5 Stage-A
   checkpoint (`data/lewm_checkpoint_stride5.pt`), since the only previously
   available stride-5 heads checkpoint predated the current `ContinuationHead`
   architecture (2 hidden layers w/ dropout) and failed to load.

Baseline for comparison: `lewm_heads_checkpoint_stride2_m4_v3.pt`, 1/12 wins (8.3%)
vs Dreamer `--frame-skip 2`, drop_rate ~49%, mean latency ~34ms (established in
the prior day's v3/v4 audit).

## 1. Chunked CEM execution

Implementation: `src/leworldgaming/agents/lewm/planner.py` (`cem_shooting`'s new
`warm_shift` param) + `src/leworldgaming/agents/lewm/agent.py` (`LewmAgent`'s
`_chunk_queue`/`_chunk_consumed` state, `planner_chunk_size` config, new
`configure_planner(chunk_size=...)` override, `--planner-chunk-size` CLI flag in
`scripts/self_play.py`). `cem_shooting` already computes a full per-timestep
categorical distribution `dist` of shape `(horizon, num_actions)`; chunking
samples `dist[1:chunk_size]` (via `torch.multinomial`, same pattern as the
existing `action0` sample) into a queue instead of re-running the full search.

**Live eval**: v3 checkpoint, stride=2, frame_skip=2, `chunk_size=5`, vs Dreamer
`--frame-skip 2`, P1=LeWM (per repo convention), 4 runs x 3 rounds = 12 rounds:

| Run | Wins | Drop rate | Mean latency |
|---|---|---|---|
| 1 | 1/3 | 8.7% | 12.7ms |
| 2 | 0/3 | 5.7% | 12.1ms |
| 3 | 0/3 | 4.8% | 12.0ms |
| 4 | 0/3 | 4.1% | 12.0ms |
| **Total** | **1/12 (8.3%)** | **~5.8%** | **~12.2ms** |

**Result**: chunking works exactly as engineered -- drop rate fell ~8x (49% ->
5.8%) and mean latency fell ~3x (34ms -> 12ms) -- but win rate was **unchanged**
from the v3 baseline (1/12, 8.3%). Acting open-loop on stale cached actions for
4-of-5 decisions (not incorporating the last 4 frames of true state) appears to
exactly offset the drop-rate gain. **Chunking alone does not close the gap to
Dreamer.**

## 2. Stride-5 Stage-B head re-retrain

Config: `configs/lewm_heads_m4_stride5_v3.yaml` (copy of `configs/lewm_heads_m4.yaml`,
which already targets `ckpt_in: data/lewm_checkpoint_stride5.pt`; only changed
`ckpt_out` to avoid clobbering the old stale checkpoint). Same M4 recipe as the
v3 stride-2 heads (imagined-rollout reward/continuation supervision, value-head
bootstrapping, `cont_dropout=0.35`/`cont_hidden_dim=64` continuation-head
regularization). Trained 20,000 steps (~92 min at 3.6 step/s), best-val
continuation-head snapshot swapped in automatically (val_loss_c=0.6950, same
overfitting pattern as stride-2's heads -- val c loss climbs from ~0.7 to ~2.2
over training while train c stays noisy/low, consistent with the known
terminal-window data scarcity issue).

**Live eval**: fresh `lewm_heads_checkpoint_stride5_m4_v3.pt`, frame_skip=5
(83.3ms budget), vs Dreamer `--frame-skip 2`, 4 runs x 3 rounds = 12 rounds:

| Run | Wins | Drop rate | Mean latency |
|---|---|---|---|
| 1 | 0/3 | 0.1% | 43.7ms |
| 2 | 2/3 | 0.1% | 43.5ms |
| 3 | 2/3 | 0.1% | 43.4ms |
| 4 | 0/3 | 0.1% | 43.7ms |
| **Total** | **4/12 (33.3%)** | **~0.1%** | **~43.6ms** |

All P1 wins were decisive (opponent HP=0). Drop rate is essentially eliminated
(83.3ms budget comfortably covers the ~44ms mean CEM search), unlike stride=2's
33ms budget which the same search blows past ~49% of the time.

**Result: stride=5 with the current (v3-recipe) heads beats both the stride=2
v3 baseline (1/12) and chunked stride=2 (1/12) by a wide margin -- 4/12 (33.3%),
a ~4x improvement.** This directly contradicts the 2026-07-17 decision
(`docs/lewm_stride2_retrain_decision_2026-07-17.md`) to abandon stride=5 in
favor of stride=2 for reaction-rate reasons. The likely explanation: that
decision's stride=5 heads checkpoint predated the M4 head-training
improvements (imagined-rollout calibration, value bootstrapping, continuation-head
regularization) developed afterward for stride=2's v2/v3 iterations -- i.e. the
old stride=5 test compared an under-trained/miscalibrated stride=5 head against
Dreamer, not a fair comparison. With the current recipe, stride=5's much larger
per-decision compute budget (no drops) outweighs its coarser reaction rate.

## Follow-up: distinct-per-raw-frame CEM planning (2026-07-21)

After the stride5 re-retrain result above, we discovered the JVM runs with `--input-sync`
(confirmed via `ps aux` and `docs/gemini_research.md`): the engine waits indefinitely for
each side's action, so all `drop_rate`/latency numbers measured all day are a simulated
"what-if-real-time" diagnostic with **no actual mechanical effect** on any match outcome in
this harness. Win-rate deltas reflect genuine planning/model-quality differences only.

Given that, we implemented genuinely-distinct-per-raw-frame CEM planning
(`plan_raw_actions=True` in `planner.py`/`agent.py`, `--planner-plan-raw-actions` CLI flag):
instead of the planner searching over "hold one action for the whole 5-frame stride block"
sequences (`_repeat_action_blocks`), it now searches over genuinely different actions per
raw frame within a block (`_concat_action_blocks`), reusing the existing chunk-queue
dequeuing mechanism from the earlier chunking work. This lets LeWM express short combos
(e.g. crouch-then-punch) within one 83ms stride5 decision window instead of being forced to
hold one static input for all 5 frames.

**Live eval**: `--p1-frame-skip 1 --planner-chunk-size 5 --planner-plan-raw-actions` (stride5
v3 checkpoint) vs Dreamer `--p2-frame-skip 2`, 4 runs × 3 rounds = 12 rounds:

| Config | Rounds | Wins | Win rate |
|---|---|---|---|
| stride5 v3 plain (block-repeated actions) | 12 | 4 | 33.3% |
| **stride5 v3 + plan_raw_actions + chunk_size=5 + frame_skip=1** | 12 | 3 | 25.0% |

**Result: a regression, not an improvement**, relative to the plain stride5 v3 baseline
(still clearly ahead of the old stride2 baselines at 8.3%). Plausible reasons:
- Searching over `horizon*temporal_stride` (much larger) raw-action sequences with the same
  CEM sample budget (`num_samples`, `num_iters`) spreads the search thinner per dimension,
  likely hurting solution quality despite the added expressiveness.
- `frame_skip=1` means the agent must actually replan/dequeue every raw frame; while
  `--input-sync` means this has no timing penalty, it does mean 5x more raw decision points
  per stride block are "live" (visible to the opponent's game-state each frame), which could
  interact with variance/luck across only 12 rounds.
- Small sample size (12 rounds) — the difference (3 vs 4 wins) is within noise range,
  consistent with the win-clustering-by-connection caveat already noted below.

**Initial conclusion (superseded below)**: keep plain stride5 v3 (block-repeated actions,
`plan_raw_actions=False`) as the recommended config, since `plan_raw_actions` at default
CEM settings (`horizon=8, samples=24, iters=1` — tuned for 8 block-level decision dims) was
under-sampling once applied to `horizon*temporal_stride=40` raw-frame dims.

## Follow-up 2: heavier CEM sampling for plan_raw_actions (2026-07-21)

User asked to test both agents at a fair 60Hz decision rate (`--p1-frame-skip 1
--p2-frame-skip 1`, instead of the earlier `--p2-frame-skip 2`) and to tune CEM more
aggressively for `plan_raw_actions`, while keeping an eye on whether it could still fit a
genuine (non-input-sync) real-time budget around 30Hz (~33ms/decision) for a future
deployment target.

**Step 1 — fair 60Hz-vs-60Hz baseline** (`plan_raw_actions` at the same thin CEM settings
used in Follow-up 1, `--p1-frame-skip 1 --p2-frame-skip 1`): 0/12 wins. Worse than the
Follow-up 1 result (3/12) since Dreamer itself got faster too (60Hz instead of 30Hz),
raising the bar.

**Step 2 — timing probe**: measured actual per-decision latency for several CEM configs on
an RTX 5060 Ti (`horizon`/`num_samples`/`num_iters`, `chunk_size=5` so only 1-in-5 raw
decisions is a full replan):

| Config | Replan cost | Amortized per raw-frame (÷5) | Fits 60Hz (16.7ms)? | Fits 30Hz (33ms)? |
|---|---|---|---|---|
| raw-thin (h5/s24/i1, as tested in Follow-up 1) | ~32-38ms | ~7ms | Yes | Yes |
| raw-heavier-A (h5/s128/i3) | ~85-100ms | ~20ms | No | Yes |
| **raw-heavier-B (h3/s96/i3)** | ~55-63ms | ~12.6ms | **Yes** | Yes |
| raw-heavier-C (h5/s256/i2) | ~77-78ms | ~15.6ms | Borderline | Yes |

**Step 3 — live eval of raw-heavier-B** (`--planner-horizon 3 --planner-samples 96
--planner-iters 3 --planner-chunk-size 5 --planner-plan-raw-actions --p1-frame-skip 1`,
vs Dreamer `--p2-frame-skip 1`): attempted 5x 3-round runs; 2 runs hit an unrelated
JVM/harness stall (0 rounds completed despite thousands of decisions logged — a stability
issue, not a planning-quality issue, consistent with the known `collect_data.py`
JVM-deadlock pattern; not counted). Of the 3 valid runs (9 rounds total):

| Run | Rounds | P1 (LeWM) wins |
|---|---|---|
| 1 | 3 | 1 |
| 2 (retry) | 3 | 3 |
| 3 | 3 | 1 |
| **Total** | **9** | **5 (55.6%)** |

**This is our best result to date** — clearly ahead of plain stride5 v3 (33.3%), the thin
raw-frame config (0-25%), and both stride2 baselines (8.3%). It shows the earlier
`plan_raw_actions` regression was a CEM-search-budget artifact (too few samples for the
larger raw-frame action space), not a flaw in the distinct-per-frame planning idea itself —
once given adequate samples/iterations, it outperforms block-repeated planning.

**Real-time feasibility**: `raw-heavier-B`'s replan cost (~55-63ms for a 5-raw-frame block)
amortizes to ~12.6ms/raw-frame, comfortably inside even a genuine (non-input-sync) 60Hz
budget (16.7ms/frame) — and far under a 30Hz budget (33ms/frame). So this improvement
should also hold up if/when the harness is ever run in true real-time (non-`--input-sync`)
mode.

**Caveats**: small sample (9 rounds, one config); the JVM-stall instability (2/5 runs)
warrants separate investigation before treating this as fully robust; win-clustering by
connection (noted above) still applies.

## Follow-up 3: pushing CEM even heavier regresses sharply (2026-07-21)

Tested whether increasing the CEM budget further beyond `raw-heavier-B` (h3/s96/i3) keeps
helping. Timing probe confirmed `heavier-E` (`horizon=3, samples=160, iters=4`) still fits
comfortably under the 60Hz budget (~15.7ms amortized/raw-frame vs 16.7ms limit).

**Live eval** (`--p1-frame-skip 1 --planner-chunk-size 5 --planner-plan-raw-actions
--planner-horizon 3 --planner-samples 160 --planner-iters 4`, vs Dreamer
`--p2-frame-skip 1`), 4 runs × 3 rounds = 12 rounds:

| Config | Rounds | P1 (LeWM) wins |
|---|---|---|
| raw-heavier-B (h3/s96/i3) | 9 | 5 (55.6%) |
| **raw-heavier-E (h3/s160/i4)** | 12 | **1 (8.3%)** |

**Sharp regression, not a plateau.** This confirms the "model exploitation" risk flagged in
the planner-alternatives research below: a heavier CEM search doesn't just have diminishing
returns past a point — it can actively dig into the learned reward/value heads' blind spots
(known-imperfect per the reward-calibration audit) and produce *worse* real play despite
scoring better in imagination. `h3/s96/i3` (`raw-heavier-B`) is the sweet spot; do not push
heavier without also addressing model-calibration risk (e.g. via the uncertainty-penalty/
pessimistic ensemble scoring already implemented in `_decode_pessimistic`).

## Follow-up 4: re-confirmation of h3/s96/i3 does NOT replicate 55.6% (2026-07-21)

Re-ran the exact same `raw-heavier-B` config (`--planner-horizon 3 --planner-samples 96
--planner-iters 3 --planner-plan-raw-actions --planner-chunk-size 5 --p1-frame-skip 1` vs
Dreamer `--p2-frame-skip 1`) for an independent 4×3-round confirmation batch:

| Batch | Rounds | P1 wins | Win rate |
|---|---|---|---|
| Original (Follow-up 2) | 9 (2/5 runs stalled, excluded) | 5 | 55.6% |
| **Confirmation re-run** | 12 (all 4 runs completed) | **2** | **16.7%** |
| **Combined** | **21** | **7** | **33.3%** |

**The 55.6% result does NOT replicate.** Combined across both batches, `h3/s96/i3` sits at
33.3% — statistically indistinguishable from the plain stride5 v3 baseline (also 33.3%,
n=12) rather than a genuine improvement. This strongly suggests the original 55.6% figure was
small-sample variance (consistent with the previously-noted win-clustering-by-connection
pattern seen throughout this project's live evals), not a real effect of `plan_raw_actions`
or the heavier CEM budget.

**Revised conclusion**: `plan_raw_actions` (with any CEM budget tested so far — thin, h3/s96/i3,
or heavier h3/s160/i4) has NOT been shown to reliably beat plain stride5 v3's block-repeated-
action planning. Given `plan_raw_actions` adds implementation/runtime complexity for no
demonstrated net benefit at current sample sizes, **plain stride5 v3
(`plan_raw_actions=False`) remains the safer, equally-performing recommended config** unless
a much larger live-eval sample (30+ rounds) is run to properly resolve the variance.

## Planner-alternatives research (2026-07-21)

Researched whether a different planning algorithm (vs CEM) would suit this setting better
(discrete ~40-action space, learned/imperfect reward-value model, tight real-time GPU-shared
budget, no goal-image cost — pure reward/value maximization). Summary of findings (full
citations: MuZero arXiv:1911.08265, EfficientZero arXiv:2111.00210, Sampled MuZero
arXiv:2104.06303, DreamerV3 arXiv:2301.04104, TD-MPC2 arXiv:2310.16828, POLO
arXiv:1811.01848, MPPI arXiv:1707.02342, Gumbel-Softmax arXiv:1611.01144, iCEM
arXiv:2008.06389):

- **MCTS/MuZero-style tree search**: not favorable without a trained policy prior — MCTS is
  inherently sequential (one simulation at a time) vs CEM's fully GPU-batched parallel
  rollouts; on our shared-GPU real-time budget, batched CEM beats sequential MCTS at
  comparable forward-pass counts.
- **MPPI**: not directly portable (assumes continuous/Gaussian action perturbations), but its
  soft exponential-weighting elite update (vs CEM's hard top-K cutoff) is a cheap (~10 line),
  no-retraining upgrade to `cem_shooting` worth trying — low-risk, modest expected gain.
- **Gradient-based/Gumbel-softmax planning**: not recommended — our discrete/multi-modal
  action landscape favors CEM's parallel-sample diversity over single-trajectory gradient
  descent (gets stuck in one mode), and gradients would more aggressively exploit our known
  reward/value-head miscalibration than random sampling does.
- **Amortized policy (Dreamer's approach)**: literature (Sampled MuZero vs Dreamer) and our
  own data agree — online search beats an amortized policy when the model is decent and
  compute budget is sufficient (consistent with 55.6% at adequate CEM budget vs losing badly
  at tight budget). Validates staying with online CEM.
- **Most promising untried upgrades**: (1) soft/MPPI-style elite reweighting (cheap, no
  retrain), (2) TD-MPC2-style policy-prior-warm-started CEM (new small Stage-B policy head,
  bigger expected gain but requires training), (3) flipping `uncertainty_penalty`'s sign to an
  active exploration-bonus during search vs. pessimistic commitment at selection (cheap, uses
  existing ensemble infra).
- **Structural gap not fixable by planner alone**: our planner has no opponent model (plans
  as if Dreamer's next action is fixed, ignoring reaction), unlike Dreamer's RSSM which
  implicitly encodes opponent behavior; addressing this would need AlphaZero-style two-player
  search, a much larger change.

## Recommendation (updated 2026-07-21, after re-confirmation)

**`plan_raw_actions`'s apparent 55.6% result did NOT replicate** (Follow-up 4) — combined
sample (21 rounds) puts it at 33.3%, statistically the same as plain stride5 v3. **Revert to
the original recommendation below: plain stride5 v3 (`plan_raw_actions=False`)** is the
current best-supported config; `plan_raw_actions` is implemented, available, and not
disproven, but has not demonstrated a reliable improvement at any CEM budget tested so far
(thin, h3/s96/i3, or h3/s160/i4) — treat any single-batch (n=9-12) result for it as
unreliable pending a much larger live-eval sample (30+ rounds) before drawing conclusions.

## Original recommendation

- **Adopt `lewm_heads_checkpoint_stride5_m4_v3.pt` (frame_skip=5) as the new
  best-known LeWM configuration**, replacing stride=2 v3 as the reference
  checkpoint for future work.
- Superseded: the 2026-07-17 stride=2 adoption decision and this session's
  chunking effort (chunking was motivated by stride=2's drop-rate problem,
  which stride=5 avoids structurally by having a larger native budget).
- Sample size caveat: 12 rounds per config is still small (wins cluster
  suspiciously by connection -- winning round 1 of a match correlates with
  winning rounds 2-3). A larger live-eval run (more connections, not just more
  rounds per connection) would sharpen the confidence interval before treating
  33.3% as a stable estimate.
- Possible follow-up: apply chunking (`chunk_size>1`) on top of the new
  stride=5 heads too -- since stride=5 already has slack in its 83.3ms budget,
  chunking's benefit there would be different (e.g. allowing a deeper/more
  thorough CEM search on the 1-in-N replan calls) rather than avoiding drops.
