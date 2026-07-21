# LeWM stride=2 vs Dreamer (frame-skip=2) — live self-play results

## Setup

Reused the existing stride=2 checkpoints (no retrain — Stage A/B already
done): `data/lewm_checkpoint_stride2.pt` + `data/lewm_heads_checkpoint_stride2_m4.pt`.
Dreamer run with `--p2-frame-skip 2` to match LeWM's 30Hz decision rate
(same "equalize decision rate" ablation methodology as the stride=5 test
in `docs/lewm_stride2_retrain_decision_2026-07-17.md`). Engine: native
Linux DareFightingICE, RTX 5060 Ti, `scripts/self_play.py`, 3 rounds/config.

## Results

| Planner config (P1=LeWM)                                   | P1 win rate |
|--------------------------------------------------------------|:----:|
| Default (`horizon=5, samples=20, iters=1, momentum=0.1, min_prob=0.05`) | 1/3 (33%) |
| Heavier (`horizon=6, samples=32, iters=2, momentum=0.15, min_prob=0.03`) | 0/3 (0%), lost by much larger HP margins |
| Lighter (`horizon=4, samples=12, iters=1`)                    | 1/3 (33%), same rounds won/lost as default |

## Conclusion

- **Don't increase CEM search depth/samples/iters** — under real
  GPU contention with Dreamer's own concurrent inference (both agents
  share the one RTX 5060 Ti), the heavier config blew the ~33ms/decision
  budget, causing missed/late decisions and a much worse loss (0/3, larger
  HP deficits) than the default. This is a stronger regression than the
  isolated-benchmark numbers in the stride=2 retrain doc predicted, because
  those benchmarks didn't account for a concurrently-inferencing opponent.
- **Lighter config performs identically to default** (same 1/3 win rate,
  same round outcomes) — the current defaults are already near the
  latency/search-depth sweet spot; there's no free win from either
  direction of planner-only tuning.
- **LeWM at stride=2 is a real improvement over stride=5** (previously
  lost every round vs live Dreamer) but still trails Dreamer's native 60Hz
  even at matched 30Hz decision rate — reaction-speed parity alone doesn't
  close the gap; Dreamer's actor/value net likely also benefits from more
  training data/steps than LeWM's Stage-B heads.
- No further planner-only lever closes this gap. Next steps (not done
  here, out of scope for a planner-tuning pass): more Stage-B training
  steps/data for the value/reward heads, or profiling per-decision wall
  time directly (no latency instrumentation currently exists in
  `LewmAgent`/`agent_vs_agent.py`) to confirm the budget-overrun theory
  above rather than inferring it from win-rate deltas alone.
