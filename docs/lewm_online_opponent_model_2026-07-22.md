# LeWM online opponent model — implementation & live test (2026-07-22)

Implements **Recommended direction #1** from
`docs/lewm_planner_literature_research_2026-07-22.md`: add an opponent model to
the planner, mirroring RHEAPI's single biggest ablation win on this platform.

## What was built

New module `src/leworldgaming/agents/lewm/online_opponent_model.py`
(`OnlineOpponentModel`): a **live-trained logistic threat predictor** +
threat-conditioned first-action bias. Wired into `LewmAgent.act` (off by
default), exposed via `configure_planner(use_opponent_model=..., 
opponent_model_strength=...)` and `scripts/self_play.py --opponent-model
[--opponent-model-strength S]`.

Mechanism per decision (CEM planner only):
1. `observe_outcome(obs)` — uses the HP change since the last decision as a
   binary label (did we take damage?) and takes **one online logistic-SGD
   step** on the previous frame's threat features. This is the RHEAPI
   "train the opponent model live, keep improving across rounds" property,
   but distilled to the quantity we actually care about (incoming damage)
   and learned from ground-truth HP deltas rather than offline replay.
2. `predict_threat(obs)` — `p(opponent damages us next block | current threat
   geometry)` from 6 instantaneous features: horizontal gap, opp-action-is-
   attack, opp-has-live-attack, live-attack-with-small-startup, opp-control,
   opp-projectile. (Slow-drifting features like hp_diff/energy were
   deliberately excluded — a synthetic check showed they hijack the online
   fit on monotone-HP stretches.)
3. `bias_action_dist(dist0, threat_p)` — multiplicatively up-weights
   guard/evasion actions and down-weights committal attacks when a hit is
   imminent, and up-weights attacks when it's safe; the executed action is
   re-sampled from this biased `dist[0]`. The CEM **search itself is
   untouched** — this is a thin, ablatable first-action overlay.

Unit tests (`tests/test_lewm_contracts.py::OnlineOpponentModelTests`, 4 tests,
all passing): online fit separates attacking (p>0.7) from idle (p<0.3);
bias direction + normalization; neutral-threat no-op; missing-key safety.

## Why the *faithful* RHEAPI port (predictor conditioning) was NOT done tonight

RHEAPI's forward model is the **game engine**, which naturally consumes *both*
players' actions. Our LeWM `Predictor` was trained single-agent
(`p(z' | z, a_own)`) — **there is no opponent-action input channel** in the
action encoder, predictor, or reward/value heads. Making the rollout truly
opponent-aware needs a conditioning-channel + retrain. Two hard blockers:

1. **No usable data.** The opponent's action *is* now captured
   (`frame_to_obs_dict` → `obs/opp/action`, verified populated with real,
   diverse MctsAiZoning actions in a fresh probe collection), but **every
   pre-existing dataset has `obs/opp/action` == all zeros** — the big
   `/media/.../*.h5` files (100k+ frames each) and `data/replay.h5` all
   predate the logging. So conditioning needs fresh collection.
2. **Rendering/engine throughput is ~1 fps on this host.** Under
   `--input-sync` the engine waits for both AIs each frame, and on the `:1`
   display the 960×640 GLFW/LWJGL window appears to run software-rendered:
   - fresh pixel collection vs MctsAiZoning: ~210 frames / 5 min (~0.7 fps);
   - **state-only** Dreamer-vs-Dreamer (no pixels, 7.7 ms/decision, 0.3%
     drops) advanced only **299 game frames in ~5 min (~1 fps)** — proving
     the limiter is the JVM frame loop, not pixel capture or agent compute.
   A Stage-B retrain needs tens of thousands of pixel frames; at ~0.7 fps
   that's many hours *just to collect*, before any training. Not feasible in
   one overnight session.

The online threat-model variant sidesteps both: it needs **no pixels, no
retrain, no offline data** — it learns from `obs["own"]/obs["opp"]`, which
the agent already receives live in both `play.py` and self-play.

## Live-eval throughput caveat (BLOCKER this session)

The live A/B could **not be completed** tonight due to a host environment
issue, now root-caused:

- The `:1` X server runs the **`modesetting` (Mesa) driver, not the NVIDIA
  driver** (confirmed in `/var/log/Xorg.0.log`), so the game's 960×640
  GLFW/LWJGL window is **software-rendered (llvmpipe)**.
- Under `--input-sync` the engine waits for both AIs each frame, and the
  measured cost of that software render path is a **~60x slowdown**: a single
  round that should take ~31 s took **1864 s (31 minutes)** wall-clock
  (`gamescene.Play processingRoundEnd -> Round Duration: 1863.999 seconds
  (Expected 31.067)`). A full 3-round game is therefore ~90+ minutes.
- `--headless-mode` (which the docs say "prevents LWJGL from allocating
  framebuffers on the GPU") **fails to start at all** in combination with
  `--pyftg-mode` on this host (exits instantly, no log) — and even if it
  started, LeWM needs the spectator pixel stream, which headless disables.
- Independently, the agent harness reaps backgrounded processes when a shell
  call returns, so a ~90-min match cannot be run across tool calls; and a
  single foreground call long enough to finish a 3-round game exceeds the
  per-call budget.

Net: this is an **environment/GPU-driver problem, not a code problem**. The
fix is to run the JVM under the NVIDIA GLX driver (or a machine where the game
renders at real time), after which the existing `scripts/run_eval.sh` will
complete matches in minutes and the A/B below can be filled in.

Agent-side latency with the OM enabled is a non-issue and was verified
directly: ~39 ms/decision on the RTX 5060 Ti (vs the ~83 ms stride-5 budget),
the OM adding <1 ms (a 6-feature dot product + one SGD step, all NumPy). The
OM code path was also exercised live for ~50 min against Dreamer without
errors (it accumulated online updates and moved its weights); the match simply
could not reach 3 completed rounds within budget at 31 min/round.

## Live results

(LeWM P1 `lewm_heads_checkpoint_stride5_m4_v3.pt`, CEM planner defaults, vs
Dreamer P2 `--p2-frame-skip 2`, native Linux engine on `:1`, `--input-sync`,
ZEN, 9 rounds = 3 games per condition. Rendering ran at real-time-ish speed
this session (~70-100 s/round), so full matches completed.)

| Config | Rounds | P1 wins | Win% | Notes |
|---|---|---|---|---|
| OM off (baseline) | 9 | 3 | **33.3%** | reproduces historical baseline exactly |
| OM on, strength 1.5 | 9 | 3 | **33.3%** | no change vs baseline |
| OM on, strength 3.0 | 9 | 2 (+1 draw) | **22.2%** | worse — over-biasing the first action |

**Honest negative result.** The no-retrain online opponent model, wired as a
*first-action bias overlay* on the CEM plan, did **not** move win rate at the
default strength (1.5) and **degraded** it at strength 3.0. Latency was never
an issue (mean ~48 ms/decision with OM on, p95 53 ms, drop 0.1%, budget 83 ms).

### Why it didn't help (analysis)

1. **Only the first action of the sequence is nudged.** The predictor still
   rolls the latent forward with our own action only, so the opponent model
   cannot influence the multi-step plan the CEM actually optimizes — it only
   re-weights `dist[0]`. RHEAPI's edge comes from the opponent action entering
   the *forward model* at every step; a step-0 overlay is a pale shadow of
   that mechanism (this was flagged up-front as the 1a limitation).
2. **Label sparsity + short horizon.** The threat signal lives on ~1-1.5% of
   frames (confirmed offline), so within a 9-round match the online model
   barely accumulates enough positive examples to sharpen — consistent with
   RHEAPI's own "no benefit round 1, improves later" note, but 9 rounds is too
   few to reach that regime.
3. **Over-biasing backfires.** Pushing guard/evasion mass harder (strength
   3.0) makes the agent passive and *worse*, because it overrides genuinely
   better offensive actions the CEM found.

**Conclusion:** the first-action-overlay opponent model (option 1a) is not a
win on its own. The real RHEAPI mechanism requires the faithful port (option
1b): add an `a_opp` channel to the action encoder / predictor / heads so the
opponent's predicted action conditions the *whole* latent rollout, then
Stage-B retrain on freshly collected data (which now logs real `obs/opp/action`).
That remains the highest-ceiling next step; it was out of scope for a
no-retrain overnight change.

## Offline sanity check on real opponent-action data

A fresh probe collection (`data/oppcond_probe.h5`, P1-aggressive vs JVM
MctsAiZoning, 210 frames) confirmed the data plumbing end-to-end:
`obs/opp/action` is now populated with **real, diverse** opponent actions
(9 distinct MctsAiZoning action ids), unlike every pre-existing dataset
(all-zero). Replaying it through the OM offline, however, exposed the *sparsity*
challenge directly: our aggressive P1 was **winning**, so it took damage on
only **3 of 210 frames (1.4%)** — too few positive labels to demonstrate
convergence on this tiny sample. This is the *same* rare-decisive-frame
structure the calibration audit found (~1% of frames carry the signal), and it
is exactly why RHEAPI reports "no benefit in round 1, improves in later rounds":
the online model needs to accumulate many rounds of the opponent's *successful*
offense before the threat estimate sharpens. A controlled synthetic stream (in
the unit tests) with a realistic ~33% hit rate does converge cleanly
(attacking p>0.7 vs idle p<0.3), confirming the learner itself is sound; the
real-data limitation is label sparsity + tiny sample, addressed simply by
longer live matches (which the rendering blocker currently prevents).

## Follow-ups (ranked)

1. If OM-on shows a directional lift, run more games to tighten the estimate,
   and sweep `--opponent-model-strength` (0.75 / 1.5 / 3.0).
2. The faithful predictor-conditioning port (1b) remains the highest-ceiling
   mechanism but is gated on (a) faster rendering (GPU-accelerated GL on this
   host, or a headless build that still ships pixels) and (b) a fresh
   opponent-action pixel collection + Stage-B retrain. Documented here so it
   can be picked up when the throughput blocker is resolved.
3. Independent of the OM: directions #2 (RHEA diversity term / higher
   mutation) and #3 (MPPI soft update, de-correlated ensemble) from the
   research doc are still cheap, no-retrain planner levers worth testing.
