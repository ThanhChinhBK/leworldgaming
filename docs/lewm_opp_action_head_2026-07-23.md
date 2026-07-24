# Real-data opponent-action head vs. proxy-based opponent model (2026-07-23)

## Motivation

The earlier `online_opponent_model.py` overlay (2026-07-22) used hand-picked
geometric proxy features (distance, relative velocity, HP deltas) fit online
via logistic regression, with **no ground-truth opponent-action labels**
because no recorded dataset paired our observations with the opponent's
*actual* chosen actions. That work was explicitly flagged as a proxy and
produced a negative live result.

This session collected a fresh dataset specifically to fix that gap: full
matches of scripted P1 policies vs. a live DreamerV3 P2, recorded with real
`obs/opp/action` labels (the opponent's true executed action ID each frame)
and pixel observations, via new recording support added to
`agent_vs_agent.py` (`_SelfDrivingAI._record_transition`,
`record_buffer`/`record_pixels` params) and a new driver script
(`scripts/collect_vs_dreamer.py`).

## Data collected

- Native Linux JVM (`Main --limithp 400 400 --grey-bg --pyftg-mode
  --input-sync`), not docker, per this phase's requirements.
- 21 `.h5` files in `/media/jeovach/Hoctap/leword-opponent/`, ~38GB total:
  - `01_mixed_v_dreamer.h5`: 20 games/57 rounds/133,542 transitions (P1=`mixed`
    scripted policy).
  - `02_*aggressive_v_dreamer.h5` (20 files): single-game collections (P1=
    `aggressive` scripted policy), due to a discovered JVM quirk (see below).
- Combined: ~267,812 transitions, 117 episodes, all with real Dreamer
  opponent-action labels (23-27 unique action IDs observed) and
  224x224x3 pixel frames.
- **JVM quirk discovered**: `RunGameRequest.game_number=N` only reliably runs
  all N games on the very first request after JVM launch; subsequent
  requests to the same session only complete 1 game (3 rounds) regardless of
  N, with no exception on either side. Workaround: loop single-game (
  `--games 1`) requests in bash instead of one large request.

## Model trained

`scripts/train_opp_action_head.py`: loads the frozen Stage-A encoder +
projector from the main LeWM checkpoint
(`lewm_heads_checkpoint_stride5_m4_v3.pt`), trains a new
`OppActionHead` (`opp_action_head.py`, same 2-layer MLP architecture as
`PolicyHead`) via cross-entropy behavior-cloning: `z_t -> a_opp[t]`
(single-frame windows). Saved separately to `data/opp_action_head.pt` —
**does not touch or overwrite the main LeWM checkpoint**, consistent with
the "must keep the LeWM checkpoint" constraint.

Training (4000 steps, batch 128, full 21-file dataset, 237,870 train /
29,942 val windows):

| step | val_loss | val_acc |
|------|----------|---------|
| 200  | 1.6108   | 58.8%   |
| 400  | 1.4504   | 61.7%   |
| **600**  | **1.4462** | **61.3%** (best, checkpointed) |
| 800  | 1.5157   | 60.5%   |
| 1200 | 1.7283   | 56.6%   |
| 2000 | 1.6870   | 60.0%   |
| 4000 | 2.0170   | 52.9%   |

Best checkpoint (step 600) reaches **61.3% top-1 val accuracy on a 56-way
action space** (vs. ~1.8% chance) — genuinely predictable signal, confirming
our own latent `z` (encoding board state) carries real information about
what the opponent is about to do. Overfits past step ~800-1000 (val_acc
degrades to ~53% by step 4000); early-stopping on best val_loss handled this
correctly.

## Wiring into the planner

Added `bias_action_dist_from_opp_prediction()` to `online_opponent_model.py`:
the same counter-mapping domain knowledge as the original threat model
(favor guard/evasion actions when opponent is predicted to attack, favor
offense otherwise) but driven by `p(opponent attacks)` = sum of the
OppActionHead's softmax over `_ATTACK_IDS`, instead of an online-fit
threat probability from instantaneous geometric features.

`agent.py`: added `load_opp_action_head(path)` (loads the separate
checkpoint, does not touch the main agent state), `use_opp_action_model` /
`opp_action_model_strength` config knobs (default off), and applied the bias
as a first-action overlay in `act()`, analogous to but independent of the
existing `OnlineOpponentModel` overlay (both can be active simultaneously;
each stacks its own multiplicative reweight before the final resample).
CLI wiring added to `self_play.py` (`--opp-action-head`,
`--opp-action-strength`) and `play.py`'s `build_agent`.

All 26 existing tests in `test_lewm_contracts.py` still pass unchanged.

## Live A/B result (negative)

Native-JVM harness, `--grey-bg` (matches training distribution), LeWM (P1,
`elite_temp=1.0` default config) vs. live DreamerV3 (P2), 5 games (15
rounds) per condition:

| condition | rounds won | win rate |
|---|---|---|
| baseline (no opponent-model overlay) | 1/12* | 8.3% |
| + OppActionHead overlay (strength=1.5) | 1/15 | 6.7% |

(*one of the 6 initially-attempted baseline games returned 0 rounds due to a
transient JVM race between overlapping processes; excluded.)

**No improvement over baseline** — matches the pattern of the earlier
proxy-based `online_opponent_model.py` result (2026-07-22), despite this
head being trained on genuine ground-truth opponent-action data rather than
a proxy. Both conditions' win rates in this session (~7-8%) are also well
below the previously-reported ~20-30% baseline from earlier docker-harness
sessions with the same checkpoint/config, suggesting meaningful run-to-run
variance between harness setups (native JVM vs docker) and/or this specific
DreamerV3 checkpoint instance, not just a strength=1.5 miscalibration for
the overlay itself.

## Interpretation

1. **61% opponent-action predictability from z is real and non-trivial**,
   but predicting *what* the opponent will do doesn't by itself translate
   into a useful *first-action bias* for the planner: the CEM search already
   imagines opponent-conditioned futures via the world model's own rollout
   (it has access to the same `z` context), so a hard multiplicative
   reweight of the first action based on a coarse "will they attack" signal
   is likely fighting the search's own (better-integrated) reasoning rather
   than adding orthogonal information.
2. Consistent with the CEM planner's core strength: it already does forward
   simulation of the *opponent's likely response*, implicitly, through the
   world model's predictor/reward head trained on real self-play/opponent
   transitions — a first-action override sitting outside that loop is
   structurally the wrong place to inject even a genuinely-good opponent
   prediction.
3. A more promising next step (not attempted this session, time-boxed): feed
   the OppActionHead's prediction *into* the CEM rollout itself (e.g.
   conditioning the imagined next-frame prediction on the most likely
   opponent action rather than marginalizing/ignoring it), rather than as a
   post-hoc first-action reweight. This is closer to the RHEAPI/MAMBA-style
   joint `a_opp` conditioning noted as "deferred, out of scope for session
   time" in the constraints.

## Status

- `use_opp_action_model` defaults to **False** — this negative result is
  documented, infra kept in place (same posture as `use_policy_prior`) in
  case a future joint-conditioning approach reuses the trained head.
- Main LeWM checkpoint (`lewm_heads_checkpoint_stride5_m4_v3.pt`) and its
  `elite_temp=1.0` default config remain the best-known configuration.
- Fresh dataset (`/media/jeovach/Hoctap/leword-opponent/`, 21 files, ~38GB,
  267K transitions with real opponent actions + pixels) remains available
  for the deferred joint-conditioning retrain or a future Stage-A/world-model
  retrain, neither attempted this session due to time constraints.

## Addendum: retraining reward/value/continuation heads on the fresh data (still negative)

Beyond the post-hoc overlay tested above, this session also directly
**retrained the reward/continuation/value heads** (Stage-B, `train_lewm_heads.py`,
same architecture/config as the current-best `..._m4_v3.pt`, `configs/
lewm_heads_m4_stride5_vsdreamer.yaml`) on the fresh 21-file real-vs-Dreamer
dataset instead of the old `/media/jeovach/New Volume/leworldgaming` data —
this is a genuinely different lever from the negative overlay results above:
it changes what the *planner's own reward/value estimates* were fit on,
using the same frozen Stage-A encoder/predictor (`lewm_checkpoint_stride5.pt`,
untouched, satisfying the "keep the LeWM checkpoint" constraint).

20,000-step training run (batch_size=16, ~3.2 step/s, ~103 min total):
saved to `data/lewm_heads_checkpoint_stride5_m4_vsdreamer.pt`.

- Reward head: val loss stable ~0.24-0.29 throughout training (healthy).
- Value head: val loss stable ~0.58-0.69 throughout (healthy, comparable
  scale to the v3 checkpoint).
- Continuation head: overfits fast as previously documented for this
  architecture (val loss climbs from 0.31 at step 500 to 2.26 by step
  20000); best-val snapshot (step 500, val_loss_c=0.3077) correctly locked
  in and swapped into the final saved checkpoint by the trainer's existing
  safeguard — same known behavior as the v3 checkpoint's training run.

### Live A/B result (negative)

Native-JVM harness, `--grey-bg`, 5 games (15 rounds) vs live DreamerV3:

| condition | rounds won | win rate |
|---|---|---|
| baseline (`..._m4_v3.pt`, old-data heads) | 1/12 | 8.3% |
| **retrained heads (`..._m4_vsdreamer.pt`, fresh-data heads)** | **1/15** | **6.7%** |

Statistically indistinguishable from both the untouched baseline and the
opp-action-head overlay result above. Latency confirmed not the cause
(P1 mean 46.2ms vs 83.3ms budget, drop_rate 0.1% — same as baseline runs).

### Interpretation

Retraining the heads on genuinely-real DreamerV3 opponent data did **not**
move the needle either. Combined with the overlay result, this suggests the
bottleneck is not simply "the heads have never seen a real Dreamer
opponent" — reward/value calibration on real matches looks healthy in
isolation (stable, sane-scale losses) but doesn't translate to more wins.
Plausible remaining explanations, none tested further this session due to
time constraints:

1. **Session-level variance dominates at n=15 rounds.** All three
   conditions tested this session (baseline, overlay, retrained-heads)
   cluster at 6.7-8.3%, well below the previously-reported ~20-30% baseline
   from earlier *docker*-harness sessions with the same original v3
   checkpoint. This strongly suggests a harness-level confound (native JVM
   vs docker, or this session's specific live DreamerV3 process/seed)
   dominates over any of the three planner/head changes tested — i.e. we
   may not have actually gotten a clean apples-to-apples comparison against
   the documented ~30% number at all today.
2. Fresh dataset is scripted-policy-vs-Dreamer (mixed/aggressive), not
   LeWM-vs-Dreamer or self-play — the reward head's association between
   states/actions and outcomes may not transfer to LeWM's own action
   distribution (different induced state distribution under a different
   policy = another train/eval mismatch, of the same "distributional
   mismatch" family flagged in `configs/lewm_heads_m4_stride5_v3.yaml`'s own
   comments about grounded-vs-imagined latents).
3. 20 games (117 episodes) may simply be too little data for the value/
   reward heads to out-generalize the old dataset's coverage, especially
   given the "aggressive" P1 policy dominates 20/21 files (only 1 file used
   `mixed`), i.e. limited diversity of P1 behavior in the new data despite
   its opponent-action-label improvement.

### Recommendation

Before drawing further conclusions, the highest-value next step would be a
**controlled same-session docker-harness re-baseline** of the original
`..._m4_v3.pt` checkpoint (no changes) to establish whether today's ~7-8%
figures reflect a harness/session confound or a genuine regression from the
previously-reported ~30% — this is the missing control needed to properly
attribute today's uniformly-negative results across all three interventions
attempted (opponent-action overlay, head retrain). Not done this session due
to time; flagged as the top follow-up.

## Addendum 2: control re-baseline confirms today's harness was the confound, NOT a real regression

Per user request, re-ran the untouched `..._m4_v3.pt` checkpoint using the
**exact original validated protocol** (`docs/lewm_policy_prior_and_soft_update_2026-07-22.md`):
docker-harness (`MODE=pixels`), `--p2-frame-skip 2`, `--character ZEN` — none
of which this session's earlier native-JVM runs used (native JVM, default
frame-skip, default character).

| run | rounds | P1 wins | win rate |
|---|---|---|---|
| control 1 | 9 | 1 | 11.1% |
| control 2 | 3 | 1 | 33.3% |
| control 3 | 3 | 2 | 66.7% |
| **aggregate** | **15** | **4** | **26.7%** |

This lands squarely back in the previously-validated 20-30% range (original:
16/54=29.6%, per-run variance 11.1%-55.6%). **Confirms the hypothesis from
Addendum 1**: today's uniformly-low 6.7-8.3% results for the opp-action-head
overlay and the vsdreamer head-retrain were an artifact of an incorrect
harness/protocol (native JVM instead of docker, missing `--p2-frame-skip 2`,
default character instead of ZEN) — **not** evidence that either
intervention regressed the agent, and **not** evidence that the retrained
heads are worse than baseline. That comparison needs to be redone under the
correct protocol before any conclusion can be drawn about the new
interventions themselves.

### Corrected next step

Re-run the opp-action-head overlay and the vsdreamer-retrained-heads
checkpoint under this exact corrected protocol (docker, `--p2-frame-skip 2`,
ZEN, multiple 9-round runs) for a fair A/B against this same-session
baseline (26.7%, n=15), before concluding anything about whether either
intervention helps, hurts, or is neutral.

## Addendum 3: extended A/B under corrected protocol (18 runs, 111 rounds total)

Ran 18 independent 3-round matches per condition (baseline `..._m4_v3.pt` vs
retrained-heads `..._m4_vsdreamer.pt`) under the corrected protocol (docker
`MODE=pixels`, `--p2-frame-skip 2`, ZEN), accumulating results as they came
in to track how the comparison stabilizes with more data:

| cumulative n (rounds) | baseline win% | retrained win% | Fisher p |
|---|---|---|---|
| 15 / 15 | 26.7% | 53.3% | (early, noisy) |
| 33 / 30 | 24.2% | 43.3% | 0.12 |
| 39 / 36 | 30.8% | 44.4% | 0.24 |
| 45 / 42 | 28.9% | 42.9% | 0.19 |
| 51 / 48 | 29.4% | 39.6% | 0.30 |
| **57 / 54 (final)** | **29.8%** | **40.7%** | **0.24** |

### Result: a real but modest, statistically inconclusive positive trend

Final tally: baseline 17/57 (29.8%), retrained-heads 22/54 (40.7%) — a
~11-point edge that held up directionally across the whole run (never
crossed back below baseline as n grew) but never reached conventional
significance (p=0.24, Fisher's exact) at this sample size. Both conditions'
aggregate rates land within/near the originally-documented ~20-30% CEM
baseline range, confirming Addendum 2's diagnosis that the corrected
protocol (not the interventions) was the dominant factor in this session's
earlier confusing results.

### Honest conclusion

- **The corrected-protocol control (Addendum 2) fully explains today's
  earlier "everything is negative" pattern** — it was a harness/protocol
  confound (native JVM + wrong frame-skip + wrong character), not a genuine
  regression from any intervention tested.
- **Retraining reward/value/continuation heads on the fresh real-vs-Dreamer
  data shows a modest positive trend (29.8% -> 40.7%) that is directionally
  consistent but not yet statistically significant** at n=57/54 rounds.
  Given the known large per-run variance for this setup (documented
  11.1%-66.7% swings run-to-run even for the *same* checkpoint), this
  would need substantially more rounds (or a paired/blocked design) to
  confirm as a true improvement rather than favorable sampling.
- No planner change (CEM variants, MCTS, opponent-model overlays of any
  kind) tested across this entire investigation has produced a
  statistically confirmed win over the ~20-30% CEM+elite_temp=1.0 baseline.
  The vsdreamer-retrained-heads checkpoint is the closest candidate to a
  genuine improvement found so far and is a reasonable next default to
  promote *provisionally*, pending more evaluation rounds, rather than a
  confirmed win.

### Recommendation

Do NOT overclaim this as a confirmed improvement. If pursuing further:
adopt `lewm_heads_checkpoint_stride5_m4_vsdreamer.pt` as a candidate
checkpoint (not yet the new default), and gather another ~50-100 rounds
(ideally interleaved single-round A/B rather than blocked 3-round batches,
to reduce within-session correlation) before deciding whether to promote it.

## Addendum 4: true 60Hz-vs-60Hz fairness test, per user request (2026-07-24)

Per explicit user request, re-tested under a genuinely speed-matched
condition instead of the historical `--p2-frame-skip 2` (Dreamer throttled
to ~30Hz) handicap used in every prior comparison in this document and in
`docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`:

- **P1 (LeWM)**: `--p1-frame-skip 1` (env requests a decision every raw
  frame, ~60Hz), `--planner-chunk-size 5 --planner-plan-raw-actions`
  (genuinely distinct actions across all 5 raw frames per stride5 block,
  not holding one action — see `planner.cem_shooting`'s `plan_raw_actions`
  doc), plus two new anti-degenerate-behavior planner levers added this
  session:
  - `--planner-idle-penalty 0.3` (existing lever, penalizes NEUTRAL/no-op)
  - **`--planner-repeat-penalty 0.3` (NEW this session)**: subtracts a
    fixed cost for any planned raw frame whose action repeats the
    immediately-preceding one (including across decision boundaries, via a
    new `self._last_executed_action` tracked in `agent.py`). Implemented in
    `planner._score_action_sequences`/`cem_shooting` (new `repeat_penalty`/
    `prev_action` params), wired through `agent.py`'s `configure_planner`,
    and `scripts/self_play.py`'s new `--planner-repeat-penalty` CLI flag.
    2 new unit tests added (`test_repeat_penalty_suppresses_repeat_of_prev_action`,
    `test_repeat_penalty_zero_is_noop`) — 28 total tests, all pass.
- **P2 (Dreamer)**: `--p2-frame-skip 1` — full 60Hz raw strength, no
  throttling at all.

### Result: 0/9 rounds (0%) — LeWM loses every round at true 60Hz-vs-60Hz

| run | rounds | P1 wins |
|---|---|---|
| 1 | 3 | 0 |
| 2 | 3 | 0 |
| 3 | 3 | 0 |

Latency: P1 mean=17.2ms (just over the 16.7ms/decision budget at frame_skip=1),
**drop rate 20%** — exactly 1-in-5 decisions (the real CEM replan every
`chunk_size=5`th decision) exceeds the budget and is dropped; the docker
harness (unlike the native-JVM `--input-sync` harness used elsewhere in this
doc) does not extend real-time deadlines, so a dropped decision is a genuine
missed/late input, not just a benign timing artifact.

### Interpretation

This reproduces the same negative finding as
`docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`'s "Follow-up 2"
(0/12 at 60Hz-vs-60Hz, before this session's idle/repeat-penalty additions)
— **adding the anti-idle and anti-repeat planner levers did not change the
outcome.** Two compounding disadvantages at this setting, neither fixed by
better score-shaping:
1. **Structural speed mismatch**: Dreamer is a single forward-pass policy
   (~8.3ms/decision here); LeWM is planning-from-scratch every replan and
   its real search cost (~57ms p95 for the actual CEM call, only amortized
   over `chunk_size=5` decisions) doesn't fit even a 5x-widened 16.7ms
   budget once Dreamer is no longer throttled down to help average things
   out.
2. **Real dropped decisions**: 20% of LeWM's decisions are simply missed
   under true real-time pressure (no `--input-sync`), which is a genuine
   competitive disadvantage separate from planning quality.

### Honest conclusion on frame-skip fairness

The `--p2-frame-skip 2` protocol used everywhere else in this document (and
across most of this project's history) is **not** a true speed-matched
comparison — it deliberately throttles Dreamer to roughly LeWM's own
stride5 decision cadence to isolate *planning/model quality* differences
from *raw decision-speed* differences. That framing is legitimate for
answering "is LeWM's world-model planning any good," but it is **not**
representative of a genuinely fair real-time fight, and LeWM has not been
shown to be competitive at true 60Hz-vs-60Hz under any configuration tested
to date (this session's idle+repeat-penalty combination included).
Confirms the user's original framing (frame_skip=2 was propping up the
comparison) was correct to question.
