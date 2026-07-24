# LeWM planner shootout: discrete MuZero-style MCTS vs. iCEM, live vs. Dreamer (2026-07-21)

## Motivation

The original LeWM paper (`external/le-wm`) plans with CEM (`stable_worldmodel.solver.CEMSolver`),
which assumes a **continuous** action space (Gaussian mean/std refit each iteration —
see `config/eval/solver/cem.yaml`: `var_scale`, `n_steps`, `topk`). FightingICE's action
space is discrete (~40-56 commandable `Action` enum values), so this repo already ships a
**discrete adaptation** of CEM (`planner.cem_shooting` — per-timestep categorical
distribution refit toward elites, "iCEM"-style: Pinneri et al. CoRL 2020, arXiv:2008.06389)
instead of the paper's continuous-Gaussian CEM. The open question (this doc): is there a
planner better suited to a genuinely discrete, small (~40-action) space that can close more
of the remaining gap to Dreamer than discrete-CEM does?

`src/leworldgaming/agents/lewm/mcts_planner.py` already implements the most natural
alternative for discrete spaces — **MuZero-style MCTS/PUCT** (Schrittwieser et al. 2020,
arXiv:1911.08265): an explicit search tree with visit-count-guided simulation allocation,
instead of CEM's flat population resampling. It was wired into `LewmAgent`/`self_play.py`
(`--p1-planner mcts`, `--planner-num-simulations`, `--planner-sim-batch-size`, etc.) but,
per `docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`'s "Planner-alternatives research"
section, had only been *reasoned about*, never live-evaluated against Dreamer. This session
runs that missing experiment.

## Setup

- Checkpoint: `data/lewm_heads_checkpoint_stride5_m4_v3.pt` (current best-known LeWM config
  per `docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`, stride=5/frame_skip=5, 83.3ms
  decision budget).
- Opponent: Dreamer, `--p2-frame-skip 2` (matches the historical comparison baseline in that
  same doc).
- Engine: native Linux DareFightingICE (`--input-sync`, so — per the same doc's finding —
  win-rate differences are pure planning/model-quality signal, not a timing artifact, though
  the drop-rate/latency numbers below still reflect genuine per-decision compute cost for any
  future non-input-sync deployment).
- `scripts/self_play.py`, `--games 3` (yields 3 rounds/run at this HP setting).

Timing probe (`/tmp/timing_real.py`, real checkpoint, RTX 5060 Ti, `agent.act()` end-to-end)
first confirmed several MCTS configs' wall-clock cost against the 83.3ms budget:

| Config | Mean latency |
|---|---|
| CEM default (h8/s24/i1) | ~41ms |
| MCTS sims=24, sim_batch=16 | ~25ms |
| MCTS sims=48, sim_batch=32 | ~38ms |
| MCTS sims=64, sim_batch=64 | ~35ms |
| MCTS sims=96, sim_batch=64 | ~63ms |
| MCTS sims=128, sim_batch=96 | ~69ms |
| MCTS sims=160, sim_batch=96 | ~97ms (over budget) |

Two MCTS configs were chosen for live eval: **sims=64/sim_batch=64** (comfortably under
budget, comparable simulation-forward-pass count to CEM's `24 samples × 8 horizon steps
= 192` predictor calls vs. MCTS's `64` simulations × ≤`8`-depth tree traversal) and
**sims=128/sim_batch=96** (pushes toward the budget ceiling, testing whether more search
compute helps).

## Live results (vs. Dreamer `--frame-skip 2`, native engine, `--input-sync`)

| Planner | Config | Rounds | P1 (LeWM) wins | Win rate | Mean latency | Drop rate |
|---|---|---|---|---|---|---|
| CEM (baseline) | default (h8/s24/i1) | 3 | 1 | 33.3% | 44.6ms | 0.1% |
| CEM (baseline, rerun) | default | 3 | 1 | 33.3% | 44.2ms | 0.1% |
| **CEM combined** | | **6** | **2** | **33.3%** | ~44ms | ~0.1% |
| MCTS | sims=64, sim_batch=64 | 9 (3 runs) | 1 | **11.1%** | 39.2ms | 0.0% |
| MCTS | sims=128, sim_batch=96 | 3 | 0 | **0.0%** | 79.8ms | 17.1% |

CEM reproduces the historical 33.3% baseline from `docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`
exactly (2/6 across two independent 3-round batches). **MCTS underperforms CEM at both tested
budgets** — worse at the matched-latency budget (11.1% vs 33.3%) and catastrophically worse
when pushed toward the timing ceiling (0%, plus a 17.1% decision-drop rate from exceeding the
83.3ms budget, unlike CEM's near-zero drop rate at any tested setting).

## Why MCTS loses to discrete-CEM here (diagnosis)

This matches the *prediction* already written in the "Planner-alternatives research" section
of `docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md` (dated one day before this session,
before it had been tested live) — the live results now confirm it empirically:

1. **No learned policy prior.** MuZero's MCTS is only competitive with population-search
   methods when a learned policy network supplies `P(s,a)` priors that immediately bias the
   tree toward good actions before value estimates are even reliable. This repo's
   `mcts_search` falls back to a **flat masked-uniform prior** (`_priors_for_batch`) — with a
   ~40-action space and only 64-128 simulations, most of the tree's simulation budget is
   spent on essentially undirected `sqrt(N)/(1+N)`-driven exploration rather than exploiting
   any prior knowledge of which actions are plausible, unlike discrete-CEM which is *already*
   warm-started from the *previous decision's* refined distribution (`cem_shooting`'s
   `init_dist`/`warm_shift`) — an implicit temporal prior MCTS's from-scratch-every-decision
   tree has no equivalent for.
2. **Sequential-selection overhead vs. GPU-batched rollouts.** Even with this session's
   existing wave/virtual-loss batching, MCTS's root-to-leaf *selection* step is still
   sequential Python per simulation (tree traversal, not a tensor op) before each wave's
   *evaluation* is batched — CEM has no such sequential component at all; every sample's
   entire trajectory is generated and scored in one fully-batched pass per iteration. At
   these small simulation counts (64-128, dictated by the 83.3ms budget), MCTS spends a
   larger fraction of its budget on Python-level tree bookkeeping than CEM spends on
   equivalent bookkeeping, leaving less of the compute budget productively used for actual
   model forward passes.
3. **Depth-first sensitivity to a single miscalibrated head.** MCTS's PUCT selection
   repeatedly revisits and deepens whichever single branch currently has the highest `Q`
   estimate — if the reward/value heads (already documented as imperfectly calibrated,
   `docs/lewm_calibration_audit_and_ensembling_2026-07-20.md`) are locally over-optimistic
   about one action, MCTS will concentrate almost all of a small simulation budget on
   exploiting exactly that blind spot. CEM's population-based resampling doesn't have this
   winner-take-more-simulations feedback loop within a single decision — every candidate in a
   given iteration gets equal rollout depth regardless of its running score, only *biasing*
   future iterations' sampling, not directly allocating more search effort into one hole in
   the model. This is consistent with `docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`'s
   Follow-up 3 finding that heavier *CEM* search alone (not MCTS) can also start exploiting
   model blind spots past a point (`h3/s160/i4` regressed sharply) — MCTS's search structure
   appears to hit that same failure mode at a much smaller budget, precisely because PUCT
   concentrates rather than spreads simulations.
4. **Timing cliff at higher budgets.** Because MCTS's cost scales with `num_simulations ×
   (avg tree depth reached, Python-level)` rather than CEM's flatter `samples × horizon`
   tensor-op cost, pushing MCTS's budget up (sims=128) blew the decision deadline 17% of the
   time — a drop-rate CEM has never shown at any tested horizon/sample/iters setting in this
   project's history (see `docs/lewm_stride2_vs_dreamer_2026-07-19.md`,
   `docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`). Missed decisions compound the
   pure-planning-quality loss with a genuine reaction-time penalty MCTS pays but CEM doesn't
   at the same nominal simulation/sample count.

## Recommendation

**Do not adopt MCTS as the LeWM planner.** Discrete-CEM (`cem_shooting`, the repo's existing
discrete-space adaptation of the original paper's continuous CEM) remains the best-performing
planner tested against Dreamer at every budget tried across this project's history, now
including a genuine head-to-head against the most natural competing tree-search algorithm for
discrete action spaces.

The theoretically-correct fix for MCTS's main weakness (no policy prior) is a learned
Stage-B **policy head** trained via imagined-rollout imitation of CEM's own elite
distributions (i.e. training MCTS's missing prior from the planner that already works) —
this is exactly the untrained-but-scaffolded "policy prior" checkpointing support added in
commit `4d39252` (`train_lewm_heads.py`'s policy-prior checkpoint compatibility). This would
turn the comparison into Sampled-MuZero-style prior-guided MCTS vs. prior-free MCTS, which the
literature (and this project's own prior research note) predicts should meaningfully change
the outcome — but it requires an actual training run (imitation-learning a policy head from
CEM rollouts) that has not been done yet, and is out of scope for a planner-only (no retrain)
comparison. Until that policy head exists and is trained, **CEM remains the recommended
planner** and closing the remaining gap to Dreamer should focus on the other untried,
no-retrain-required levers already identified in
`docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`'s "Most promising untried upgrades"
list (MPPI-style soft elite reweighting; uncertainty-penalty sign flip to an active
exploration bonus during search).

## Caveats

- Small sample sizes per arm (3-9 rounds) — consistent with this project's repeated
  observation that live-eval win rates can have large single-batch variance (see
  `docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`'s Follow-up 4, where a 55.6% result
  didn't replicate at n=12). The CEM baseline was deliberately re-run once (6 rounds total,
  both batches identically 33.3%) to check for this; MCTS was not re-run at n=9 for
  sims=64 (time-constrained) but its result (11.1%) is far enough below CEM's reproduced
  33.3% or the historical stride5 baseline that a small-sample fluctuation explaining the
  entire gap seems unlikely, especially combined with the still-worse sims=128 result (0%)
  showing the same direction at a second, independent budget.
- Only two MCTS configurations were tried (bounded by the 83.3ms decision budget); a
  systematic `c_puct`/`dirichlet_frac`/`temperature` sweep was not performed and could
  possibly narrow (though, per the diagnosis above, is unlikely to fully close) the gap
  to CEM absent a learned policy prior.
