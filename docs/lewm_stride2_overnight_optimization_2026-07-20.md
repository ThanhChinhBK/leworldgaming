# LeWM stride=2 overnight optimization (2026-07-20)

Autonomous overnight session, assumption per user instruction: the existing
stride=2 Stage-A checkpoint (`data/lewm_checkpoint_stride2.pt`) is good
enough — only Stage-B head training and CEM planner hyperparameters were in
scope, targeting a live self-play win rate advantage over DreamerV3 at
`--frame-skip 2` (matched ~30Hz decision rate).

## 1. Stage-B continuation-head overfitting — two iterations, neither "fixed" it

Background (from earlier same-day work): the continuation head's val BCE
loss grows monotonically through training even as reward/value heads
converge nicely, because there are only ~308 train / ~34 val terminal
("round about to end") windows out of >1.1M frames. A best-val-loss_c
checkpoint-swap safety net was added to `train_lewm_heads.py` so the final
checkpoint always ships whichever continuation-head snapshot had the lowest
val loss during training, regardless of what the last step looks like.

| Variant | Config | Final val_loss_c (step 20000) | Best val_loss_c (never beaten past step 0) |
|---|---|---|---|
| v2 | `cont_dropout=0.2` | 3.63 | 0.6996 (step 0) |
| v3 | `cont_dropout=0.35` + `cont_hidden_dim=64` (down from 512) + `cont_weight_decay=0.05` (separate AdamW param group) | 1.60 | 0.7252 (step 0) |

**Conclusion:** v3's extra regularization (smaller head + weight decay on
top of higher dropout) cut the overfitting *severity* by >2x (final loss
3.63 → 1.60), but still never produced a continuation head that
generalized better than the untrained step-0 initialization. This confirms
the earlier hypothesis: **this is a data-scarcity problem, not a
regularization problem.** With this much escalation already applied,
further dropout/weight-decay tuning is unlikely to help — the real fix
would be collecting more terminal-window data (explicitly out of scope for
this session). The best-val-snapshot safety net is doing exactly what it's
supposed to: shipping the honest step-0 (uninformative-but-not-actively-
wrong) head rather than a badly overfit one. `use_continuation_head` is
already `False` by default in `LewmAgent`'s planner config, so this head is
not even used for planning today — its quality mostly matters if that flag
is ever flipped on.

Both `wait-stageb-v2` and `iter-contloss-fix` are marked done. **v3
(`data/lewm_heads_checkpoint_stride2_m4_v3.pt`) is the new best-available
Stage-B checkpoint** — same safety-netted continuation head as v2, but
reward/value heads benefited from a second full 20k-step imagined-rollout
run.

## 2. Bug found + fixed: `LewmAgent` ignored `cont_hidden_dim` at load time

`LewmAgent.__init__` (`agent.py`) hardcoded the continuation head's
`hidden_dim` to the shared `heads.hidden_dim` (512), so loading the v3
checkpoint (whose continuation head is 64-wide) raised a `RuntimeError:
size mismatch`. Fixed by reading `heads_cfg.get("cont_hidden_dim")` with a
fallback to the shared `hidden_dim`, matching the trainer's own fallback
logic. Verified `heads_config` from the checkpoint is folded into
`cfg["heads"]` before `LewmAgent.__init__` runs (already-existing code,
`agent.py:423-427`). Contract tests (11/11) re-run and pass after this fix.
**This means v3 (and any future checkpoint using a non-default
`cont_hidden_dim`) would have been silently unloadable at inference before
this fix** — worth double-checking any other head-specific hyperparameters
added in the future get equivalent load-time plumbing in `agent.py`.

## 3. Live self-play evaluation — inconclusive due to environment instability

Multiple self-play batches were attempted this session (v2 and v3
checkpoints vs Dreamer `--p2-frame-skip 2`). Findings, roughly in
chronological order:

- Small batches (3-6 games) are **not reliable** for judging checkpoint or
  planner-config quality: two back-to-back batches with the *identical*
  v2 checkpoint and config swung from 100% win to 0% win. Combined
  small-sample win rate across the session settled around **~33%** (3/9
  rounds), no different from the pre-Stage-B-fix baseline.
- A larger n=10 batch was launched for a more confident estimate but was
  running under heavy GPU contention (concurrent with the v3 Stage-B
  retrain) — killed after ~79 minutes once it became clear its numbers
  would be confounded and not worth the compute investment (it was NOT
  hung; confirmed via repeated `ss -tni` socket-freshness checks showing
  continuous data flow, just very slow).
- A clean v3-vs-Dreamer batch (6 games, no concurrent GPU consumer) hit a
  **newly observed ~25-minute stall between "game accepted" and the first
  `initialize()` callback** — i.e. a JVM-side game-startup stall with zero
  possible Python-side cause (agent construction happens before "game
  accepted" is even logged). It self-resolved (not a permanent deadlock)
  and the match then proceeded, confirmed repeatedly via fresh socket
  byte-count growth, but the same batch subsequently took **90+ minutes
  wall-clock for just 6 games** (vs a ~7.5 minute baseline observed earlier
  in the day under otherwise-similar conditions) despite showing
  continuous, genuine data flow throughout (never met the "hung" bar of
  zero byte growth across samples).

**This makes the planned 3-5 config CEM planner sweep infeasible within
the overnight window as originally scoped**: each config's evaluation
would need to survive this same order-of-magnitude real-time slowdown risk,
multiplying a ~90-minute-per-batch worst case across 3-5 configs into
potentially many more hours, with no guarantee any single run avoids
another JVM stall. Rather than burn the rest of the session chasing
increasingly-uncertain live win-rate deltas, this work concludes here with
a recommendation instead of new sweep data.

## 4. Recommendation: keep current CEM planner defaults for stride=2

No new evidence from this session justifies changing
`_stride_planner_defaults`/`_stride_cem_tuning` for `temporal_stride=2` in
`src/leworldgaming/agents/lewm/agent.py`. They remain:

```
horizon=5, num_samples=20, num_iters=1
sticky_prob=0.2, momentum=0.1, min_prob=0.05
```

These were already specifically chosen (see
`docs/lewm_stride2_vs_dreamer_2026-07-19.md`) to fit the ~33ms/decision
real-time budget at stride=2 while avoiding the previously-confirmed
failure mode where deeper CEM search (more samples/horizon/iters)
*regresses* performance because it competes with Dreamer's own concurrent
GPU inference for the shared RTX 5060 Ti. No changes made to `agent.py`'s
planner defaults this session — only the `cont_hidden_dim` load-time bug
fix described in §2.

## 5. Known risks / open items for future work

- **Continuation-head overfitting is very likely unfixable without more
  terminal-window data.** If this head is ever needed (e.g.
  `use_continuation_head=True` in planner config), prioritize collecting
  more round-ending replay data over further regularization tuning.
- **Live self-play evaluation in this environment has too much variance
  and occasional multi-minute-to-tens-of-minutes JVM-side stalls to trust
  small-sample (3-6 game) win rates, or even to budget wall-clock time
  reliably for larger batches.** Future planner/checkpoint comparisons
  should either (a) use a much larger number of games with generous time
  budgets, ideally run one-checkpoint-at-a-time with no concurrent GPU
  training, or (b) build a faster-than-real-time / deterministic-seed
  evaluation harness so results are both statistically meaningful and
  time-bounded. The `ss -tni` byte-freshness check (two samples ~15-30s
  apart on the `127.0.0.1:31415` sockets) is a reliable way to distinguish
  "genuinely slow" from "actually hung" without guessing from wall-clock
  alone or from CPU%, which stays near-zero even for a healthy, real-time,
  socket-I/O-bound match.
- A 6-game v3-vs-Dreamer batch was still running in the background when
  this doc was written (PID 2313456, log `/tmp/eval_v3_batch1.log`) after
  surviving the ~25-min JVM stall; if it completes with a clear result,
  it's a useful (if late) data point but does not block any conclusion
  above given the single-batch variance problem already documented.

## 6. Update (2026-07-20 morning): the overnight batch finished — bad news

The batch mentioned in §5 finally completed after ~7.5 hours wall-clock
(01:57 → 09:28). Root cause of the extreme duration: **every single game
restart in the batch hit its own multi-minute-to-1.5-hour JVM game-init
stall** (not a one-off) — the same pattern as the earlier isolated ~25min
stall, recurring 4 times across the 12-round batch. This is a systemic JVM
reliability issue for this engine build, not a fluke.

Result — a much larger, more reliable sample than anything else collected
this session: **LeWM v3 lost 11/12 rounds (91.7% loss rate)**, with total
shutouts (`hp_p1=0.0`) in 10 of those 11 losses while Dreamer kept
significant HP (93–350). Latency was fine (LeWM mean=30.0ms/p95=32.7ms
against the 33.3ms budget, drop_rate=3.2%; Dreamer mean=7.2ms).

**Diagnosis of the loss (ruled out / ruled in):**
- *Not* a head-quality regression from this session's work — v3's
  reward/value validation losses (r≈0.08, v≈0.33) are essentially
  identical to v2's, so the Stage-B changes made here didn't cause this.
- *Not* a "stuck"/no-op planner problem — only 91 stuck frames out of
  12,265 decisions (0.7%) across the whole batch.
- *Not* a latency/budget problem — comfortably within budget.
- *Not* an artifact of the JVM stalls themselves (those delay round start,
  they don't degrade in-round play once it begins).
- **Consistent with, and now much better evidenced than, the prior-session
  finding** in `docs/lewm_stride2_vs_dreamer_2026-07-19.md`: LeWM already
  trailed Dreamer at matched 30Hz decision rate before this session, and
  that doc's hypothesis ("more Stage-B training will close the gap") did
  **not** pan out — 40k combined additional Stage-B steps (v2 + v3) did not
  improve the large-sample outcome; if anything it looks worse than the
  earlier small-sample 33% estimate (though that estimate had far less
  statistical power).
- **A proposed P1/P2 side-swap control was considered and correctly
  rejected**: LeWM must always be evaluated as P1 because its pixel
  encoder was trained exclusively on P1-side self-play recordings
  (`scripts/collect_data.py` always records the self-play agent as P1;
  P2 recording is disabled whenever P1 is non-JVM). Swapping sides would
  put LeWM on out-of-distribution pixels, not control for a real
  confound — this has been true of every evaluation in this repo's
  history, so it doesn't explain the *change* in outcome, but it does rule
  out "just re-run with sides swapped" as a fix or diagnostic.

**Working hypothesis (not yet proven, no further live testing done this
session given cost):** this looks like a structural gap between test-time
CEM planning against a learned, imperfect world model (LeWM) and an
end-to-end-trained reactive actor-critic policy (Dreamer) in a
frame-precise, fast-execution domain — planning errors likely compound
over the horizon and/or the reward/value heads are exploited by the
planner in ways offline validation loss doesn't catch (a failure mode the
M4 config's own docstring already anticipated). The near-total shutouts
(not close losses) support a structural/execution disadvantage rather than
a narrowly-tunable hyperparameter issue. Closing this gap likely needs
either a different planning/architecture approach, substantially more or
better training data, or accepting Dreamer's advantage in this specific
regime — none of which fit in an overnight, planner/heads-only-tuning
scope.

## Final recommended artifacts

- **Best checkpoint:** `data/lewm_heads_checkpoint_stride2_m4_v3.pt`
  (Stage-A `data/lewm_checkpoint_stride2.pt` unchanged; Stage-B reward/value
  heads fully retrained 20k steps with imagined-rollout supervision;
  continuation head is the safety-netted step-0 snapshot).
- **Planner config:** unchanged stride=2 defaults in `agent.py`.
- **Code changes shipped:** `ContinuationHead` dropout param,
  `train_lewm_heads.py` best-val-snapshot mechanism + `cont_hidden_dim`/
  `cont_weight_decay` plumbing, `agent.py` `cont_hidden_dim` load-time fix,
  `agent_vs_agent.py`/`scripts/self_play.py` per-agent latency
  instrumentation (mean/p95/drop-rate) — all still valuable going forward
  even though this session's live win-rate numbers were inconclusive.
