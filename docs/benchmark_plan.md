# Benchmark Plan: CEM + World Models on FightingICE

This doc captures the plan for benchmarking four CEM (Cross-Entropy Method)
configurations on the FightingICE environment, and the data-collection /
training-phase changes needed to support it.

The planner implementation itself is **out of scope** here — this doc only
covers what must be in place *before* the planner is written, so that all
four arms expose a comparable interface and the resulting numbers are fair.

---

## 1. The four arms

All four use the same CEM planner with the same hyperparameters. Only the
forward-rollout dynamics differ.

1. **CEM + true game simulator** — upper bound; a custom Java AI that calls
   the built-in `simulator.Simulator.simulate()` method for rollouts
   (see §1.1 below).
2. **CEM + LeWM** — JEPA encoder + AR predictor in latent space, plus
   reward / value / continuation heads bolted on (Stage-B trainer below).
3. **CEM + Dreamer** — DreamerV3 RSSM, reward + value + continuation
   heads trained end-to-end.
4. **CEM + PETS** — ensemble dynamics on physical state, analytic reward.

### Why CEM instead of MCTS

The original LeWM paper uses CEM for planning. CEM is simpler than MCTS
(no tree, no UCB, no expansion policy), has fewer hyperparameters
(`candidates`, `elites`, `iterations`, `horizon`), and directly measures
forward-model quality: "given a budget of N rollouts of length H, how good
is the best action sequence your model can find?" Any difference in axis A
is then attributable to the world model, not tree-search details.

### 1.1 Arm 1: true game simulator (Java CEM AI)

FightingICE includes a built-in `simulator.Simulator` class (in
`src/simulator/Simulator.java`) that steps the **real game physics** —
hit detection, character state, projectiles, pushback — without rendering.
It is accessible to Java AIs via `gameData.getSimulator()`.

```java
// Simulator API (already in FightingICE)
public FrameData simulate(FrameData frameData, boolean playerNumber,
    Deque<Action> myAct, Deque<Action> oppAct, int simulationLimit)
```

`SimFighting` (the internal implementation) extends `Fighting` — it runs
the exact same physics loop as the live game, including `processingFight`,
`calculationHit`, `updateAttackParameter`, and `updateCharacter`.

**Note on existing JVM baselines:** `MctsAi23i.jar` and `MctsAiZoning.jar`
already use this simulator, but they are extremely weak:
- MctsAi23i: MCTS with **23 iteration cap** and **16.5ms time budget**
- MctsAiZoning: same structure with positional heuristics
- Both placed last in the 2025 DareFightingICE competition

For Arm 1 we need a custom Java CEM AI with proper rollout depth:

```
CEM parameters (shared across all arms):
  horizon        = 100 frames (~1.67s of game time)
  candidates     = 200 action sequences per iteration
  elites         = 20 (top-k to keep)
  iterations     = 5 CEM refinement iterations
  opponent_model = random actions (same assumption for all arms)

Per decision:
  200 candidates × 5 iterations = 1000 simulate() calls
  Each simulate(100 frames) ≈ <1ms (pure physics, no rendering)
  Total ≈ ~1 second per decision
  With --input-sync the game waits — no time pressure.
```

Arm 1 is implemented as a Java AI JAR placed in `data/ai/`, compiled
against the FightingICE classpath. Arms 2–4 are Python pyftg AIs that
run the same CEM loop but roll out through their respective world models.

### Why fixed shared planner hyperparameters

CEM knobs (`candidates`, `elites`, `iterations`, `horizon`) affect strength
independent of the world model. Tuning them per-arm conflates "model
quality" with "compute budget." Sweep once on a held-out subset, freeze a
single config, use it for all four arms. Any difference in axis A is then
attributable to the world model, not the planner.

---

## 2. Evaluation: 3-axis panel (not one number)

A single win-rate number is not informative — Dreamer/PETS are optimized
end-to-end for control while LeWM is optimized for representation. Report
three axes:

| Axis | Metric | Output space | What it measures |
|------|--------|--------------|------------------|
| **A. Planning win-rate** | win % and HP-differential vs fixed JVM opponents (`MctsAi23i`, `MctsAiZoning`), seeded | game outcomes | end-to-end control |
| **B. k-step output prediction** | reward + termination error at `k = 1, 5, 10, 20`, decoded to physical units (HP-delta; Bernoulli) | physical units | dynamics + head quality |
| **C. Probe R²** | linear probe of latent → HP_self, HP_opp, energy_self, energy_opp, x_diff, y_diff | per-model, same target | representation quality |

Crucial: axis B is in **observable space, not latent space**. Latent-space
MSE is not comparable across models because LeWM (192-d JEPA), Dreamer
(RSSM), and PETS (physical state vector) live in different spaces with
different geometry and units.

### Optional fourth axis (if useful)

- **Data efficiency** — performance vs replay size (subsample replay,
  retrain, re-evaluate axis A).

### Framing the result

The expected outcome is *not* "LeWM wins" — it's a tradeoff:

- LeWM likely wins axes B and C (its training objective).
- Dreamer / PETS likely win axis A (their training objective).
- The benchmark documents the cost of decoupling representation
  pretraining from end-to-end control.

If LeWM badly loses axis A despite winning B and C, that motivates a v2
with **value-shaping** during JEPA training (Sobal 2025) so the embedding
space is value-equivalent.

---

## 3. Data collection — changes needed

### Already in `data/replay.h5` — sufficient
- `pixels` (LeWM input)
- `action` (agent's action — see verification below)
- `reward` (HP-delta — used for axis B reward target and reward head training)
- `done`, `episode_starts` (axis B termination target and continuation head training)
- `obs/own/*`, `obs/opp/*`, `obs/global/*` (PETS / Dreamer state inputs and probe targets)

### Verify / add
1. **Opponent action**: confirm `f["action"]` is the agent's action only.
   If opponent's action is needed for axis B (opponent-action prediction),
   add `f["opp_action"]` to `scripts/collect_data.py`. Otherwise drop
   opponent-action from axis B and rely on reward + termination only —
   also valid.
2. **Eval-split tagging**: reserve a held-out set of episodes collected
   against the fixed axis-A opponents and exclude them from training. Either
   add `f["split"]` (`"train"` | `"eval"`) or write to a separate
   `data/replay_eval.h5`. Without this, axis A and axis B leak training
   data into evaluation.

### Not needed at this phase
- new observation fields beyond what's listed above
- reward shaping

---

## 4. Training phase — changes per model

### 4.1 LeWM — biggest change

Two-stage training. **Stage A is the existing `train_lewm.py` — do not
modify it.** All new work is in a Stage-B trainer.

**Stage A (existing)**: encoder + projector + action_encoder + predictor +
pred_proj, trained with JEPA prediction loss + SIGReg. Produces
`data/lewm_checkpoint.pt`.

**Stage B (new `src/leworldgaming/training/train_lewm_heads.py`)**:
- Load Stage-A checkpoint, **freeze** all JEPA components.
- Train four small MLP heads on `data/replay.h5`:
  | Head | New file | Signature | Loss |
  |------|----------|-----------|------|
  | `RewardHead` | `agents/lewm/reward_head.py` | `(z, a_oh) → twohot logits` | CE on twohot HP-delta (TD-MPC2 / DreamerV3 style) |
  | `ContinuationHead` | `agents/lewm/continuation_head.py` | `z → logit` | BCE against `1 - done` |
  | `ValueHead` | `agents/lewm/value_head.py` | `z → twohot logits` | CE on twohot λ-return with EMA target net |
  | `LinearProbe` (existing) | `agents/lewm/probe.py` | `z → 4 physical signals` | MSE — **currently random; this finally trains it** |
- Add **imagined-rollout consistency**: apply reward + continuation losses
  on predictor-rolled latents `ẑ_{t+k}` for `k = 1..H`, not only on
  encoder-produced `z_t`. CEM uses rolled latents at inference, so heads
  must work there.
- Extract shared replay-loading utilities (`_valid_seq_start_indices`,
  `_sample_sequence_batch`, `_to_device_seq`) from `train_lewm.py` into
  `src/leworldgaming/training/_replay_utils.py` first; both trainers
  import from there.
- Extend checkpoint dict and `Agent.load_state_dict`
  (`src/leworldgaming/agents/lewm/agent.py`) to include the four heads.

**New config block in `configs/lewm.yaml`**:
```yaml
heads:
  reward_bins: 41
  value_bins: 41
  hidden_dim: 512
  freeze_jepa: true
  imagined_horizon: 5
  lambda_return: 0.95
  target_ema: 0.99
  reward_loss_weight: 1.0
  cont_loss_weight: 1.0
  value_loss_weight: 0.5
```

**Implementation order within Stage B**:
1. Extract replay utils.
2. Add the three new head modules.
3. Train reward + continuation only (no bootstrap) — sanity check.
4. Add value head with TD(λ) and EMA target.
5. Add imagined-rollout consistency.
6. Validate via reward-head R² on val split and via the existing
   CEM planner now using a real reward signal instead of a random probe.

### 4.2 Dreamer — minimal change

DreamerV3 (vendored at `external/dreamerv3-torch/`) already trains reward,
value, and continuation heads as part of its standard loss.

- Train to convergence with `src/leworldgaming/training/train_dreamer.py`.
- Save a checkpoint to `data/dreamer_checkpoint.pt`.
- (Planner phase, not now) Unstub the online `act()` path so MCTS can call
  `(z, a) → z'`, `r̂`, `ĉ`, `V̂`.

### 4.3 PETS — minimal change

PETS already trains ensemble dynamics on physical state and uses an
analytic HP-delta reward.

- Train to convergence with `src/leworldgaming/training/train_pets.py`.
- Save a checkpoint to `data/pets_checkpoint.pt`.
- Optional: small continuation head (otherwise infer round-end from
  decoded state's HP).

### 4.4 Shared probe training (axis C fairness)

Add `src/leworldgaming/training/train_probes.py`: fits **the same**
`LinearProbe` architecture against **the same** physical target
(HP_self, HP_opp, energy_self, energy_opp, x_diff, y_diff) on each
model's latent. R² is then directly comparable across models.

For LeWM the probe is already trained inside Stage B; for Dreamer / PETS
the probe is trained post-hoc on a held-out probe-training split.

---

## 5. Order of work (training phase)

1. Verify and patch `replay.h5` schema (opponent action; eval split).
2. Train Dreamer + PETS to convergence in parallel — no code changes.
3. Implement LeWM Stage-B trainer (the largest piece).
4. Train shared linear probes for axis C.
5. (Separate later phase) MCTS planner, env wrapper for arm 1, tournament
   harness in `src/leworldgaming/eval/tournament.py`.

After step 4, every world model exposes a comparable interface
(`(z, a) → z', r̂, ĉ, V̂`) and a trained shared probe — the benchmark is
ready to run as soon as the planner exists.

---

## 6. References

- MuZero — latent dynamics + reward / value / policy heads with MCTS:
  https://arxiv.org/pdf/1911.08265
- TD-MPC2 — twohot reward / value, EMA target, latent MPC:
  https://arxiv.org/abs/2310.16828
- DreamerV3 — twohot, λ-return, target net (vendored in this repo):
  https://arxiv.org/abs/2301.04104
- Value-guided JEPA planning (Sobal 2025) — value-shaped JEPA, motivates
  the v2 path:
  https://arxiv.org/abs/2601.00844
- DINO-WM — JEPA-style world model with MPC planning:
  https://dino-wm.github.io/
- LeWorldModel project page:
  https://le-wm.github.io/
