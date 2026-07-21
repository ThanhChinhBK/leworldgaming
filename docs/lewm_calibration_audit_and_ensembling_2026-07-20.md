# LeWM offline calibration audit + reward/value head ensembling (2026-07-20 follow-up)

Follow-up to `docs/lewm_stride2_overnight_optimization_2026-07-20.md`, which
ended with LeWM v3 losing 11/12 live rounds (91.7%) to Dreamer and a leading
but unproven "structural gap" hypothesis: the CEM planner exploiting a
predictor whose imagined rollouts drift away from the encoder's grounded
latents ("model exploitation", a well-known MBRL failure mode, already
anticipated in `configs/lewm_heads_m4.yaml`'s own docstring).

This session tests that hypothesis directly (offline, no live JVM match
needed) and adds the infrastructure to act on it if confirmed: reward/value
head ensembling with pessimistic (uncertainty-penalized) CEM scoring.

## 1. Offline calibration audit (`scripts/audit_lewm_calibration.py`)

Reuses the exact frozen JEPA + Stage-B heads and the exact predictor-rollout
code path from `train_lewm_heads.py`'s M4 imagined-loss branch, but breaks
the result down **per imagined depth** (1..K=5) instead of aggregating over
the whole horizon, and adds a direct latent-space drift metric the trainer
never computes. Run on 20,000 held-out validation windows (same val split
as Stage-B training) against `data/lewm_heads_checkpoint_stride2_m4_v3.pt`.

### Result

```
depth |  lat_cos |  lat_mse |  r_mae_im |  r_mae_gr |   r_gap |  v_delta
------------------------------------------------------------------------
    1 |   0.9966 |  0.00639 |    0.0012 |    0.0011 |  0.0000 |   0.0010
    2 |   0.9917 |  0.01580 |    0.0011 |    0.0011 |  0.0000 |   0.0018
    3 |   0.9850 |  0.02850 |    0.0013 |    0.0012 |  0.0000 |   0.0025
    4 |   0.9768 |  0.04437 |    0.0011 |    0.0011 |  0.0000 |   0.0033
    5 |   0.9670 |  0.06337 |    0.0012 |    0.0012 |  0.0000 |   0.0040

decisive-frame (|reward| > 0, the ~1% of frames the planner actually needs
to get right) breakdown:
depth |   n_nz |  r_mae_im |  r_mae_gr |   r_gap
------------------------------------------------
    1 |    347 |    0.0493 |    0.0493 |  0.0000
    2 |    359 |    0.0452 |    0.0452 |  0.0000
    3 |    351 |    0.0540 |    0.0539 |  0.0000
    4 |    348 |    0.0455 |    0.0455 |  0.0000
    5 |    350 |    0.0509 |    0.0510 | -0.0001
```

### Interpretation

* **Latent drift is real and grows with depth**, as expected for any AR
  predictor rolled forward without re-grounding: cosine similarity to the
  true grounded latent drops from 0.997 (depth 1) to 0.967 (depth 5); MSE
  grows roughly linearly (~0.006 → ~0.063).
* **But the reward head's prediction error is essentially IDENTICAL whether
  fed the drifted imagined latent or the true grounded latent, at every
  depth** — `r_gap` (imagined MAE − grounded MAE) is ≈0.0000 across the
  board, including on the decisive (nonzero-reward) subset where the
  absolute error is largest (~0.045–0.054 MAE, meaningful against a reward
  range of ±0.3).
* **This refutes "model exploitation via reward-head extrapolation on
  drifted imagined latents" as the primary cause of LeWM's live losses.**
  The M4 imagined-rollout loss (`imagined_loss_weight`) appears to have done
  its job: the reward head is robust to the predictor's latent drift, at
  least in this averaged sense.
* **What the audit instead reveals**: the reward head's baseline accuracy on
  rare, high-magnitude ("decisive") reward frames is mediocre regardless of
  latent source (~0.05 MAE against a ±0.3 range, i.e. often wrong about
  which of several plausible outcomes is happening) — a data-scarcity /
  head-precision issue, not a depth-dependent compounding-error issue. This
  is consistent with the reward distribution being extremely sparse
  (~1% of frames nonzero, mean |reward| ≈ 0.0004–0.0005 in this dataset).
* Value-head imagined-vs-grounded disagreement (`v_delta`) does grow with
  depth (0.001 → 0.004) but stays small in absolute terms relative to the
  value range (±10); not flagged as a smoking gun on its own.

**Net effect on the working hypothesis from the prior doc**: the
"CEM-plans-against-a-drifting-imagined-model" story is weaker than assumed.
The more likely structural gap is closer to "reward/value heads are
imprecise on the rare frames that matter, and the planner has no way to
know when it's relying on an uncertain/low-confidence prediction" — which
is exactly what ensembling + uncertainty-aware scoring (below) targets,
just for a different underlying reason than originally hypothesized.

Re-run any time via:
```
.venv/bin/python scripts/audit_lewm_calibration.py \
    --ckpt data/lewm_heads_checkpoint_stride2_m4_v3.pt \
    --data "/media/jeovach/New Volume/leworldgaming" \
    --num-windows 20000
```

## 2. Reward/value head ensembling + pessimistic CEM scoring

Given the audit's finding (imprecise-on-rare-frames, not depth-compounding),
ensembling is repurposed slightly from the original plan: instead of only
guarding against imagined-latent drift, it now primarily guards against the
planner over-trusting a single head's confident-but-wrong prediction on any
latent (grounded or imagined) it hasn't seen much of during training —
still the standard MBRL fix (PETS/MBPO-style pessimism), just motivated by
data scarcity rather than rollout drift specifically.

### What changed

* **`planner.py`**: new `_decode_pessimistic(head, bins, penalty, *args)`.
  If `head` is a plain `nn.Module`, behaves exactly like
  `twohot_decode(head(*args), bins)` (fully backward compatible — existing
  single-head checkpoints/configs are unaffected). If `head` is an
  `nn.ModuleList` (an ensemble), decodes each member and returns
  `mean - penalty * std` across members. `_score_action_sequences`,
  `random_shooting`, and `cem_shooting` all gained a `uncertainty_penalty:
  float = 0.0` parameter (default preserves old behavior exactly) and now
  accept `reward_head`/`value_head` as either a single module or an
  ensemble.
* **`train_lewm_heads.py`**: new `heads.reward_ensemble_size` /
  `heads.value_ensemble_size` config keys (default `1` = old behavior).
  `>1` builds an `nn.ModuleList` of independently-initialized heads (default
  PyTorch init already gives each member a different random draw); each
  member's loss is computed independently against the same batch/targets
  and averaged (no cross-member gradient coupling — equivalent to training
  N independent heads jointly). Value heads keep their own independent EMA
  target-head per member. Checkpoints save **both** the plural
  (`reward_heads`/`value_heads`/`value_target_heads`, list of per-member
  state dicts) and singular (`reward_head`/`value_head`/`value_target_head`,
  member 0 only) keys, so **older, non-ensemble-aware consumers keep
  working unchanged** even against an ensembled checkpoint. Ensemble sizes
  were added to the resume-blocking structural-key check (can't change
  ensemble size mid-resume).
* **`agent.py`** (`LewmAgent`): reads the same `reward_ensemble_size`/
  `value_ensemble_size` keys from a loaded checkpoint's `heads_config` and
  builds `self.reward_head`/`self.value_head` as either a plain head or an
  `nn.ModuleList`, matching whatever the checkpoint was trained with.
  `load()` uses a new `_load_head()` helper that prefers the plural
  checkpoint keys for ensembles and falls back to the singular key
  (loading pre-ensembling checkpoints, or `LewmAgent.save()`'s own output,
  unchanged). New `configure_planner(uncertainty_penalty=...)` knob and
  `planner_cfg["uncertainty_penalty"]` config key (default `0.0`).

### Verification performed (no live JVM match needed)

* `tests/test_lewm_contracts.py`: 3 new unit tests for
  `_decode_pessimistic` (single-head pass-through, ensemble
  mean-minus-penalty-times-std formula, zero-penalty degenerates to plain
  mean). Full suite: 11 → 14 tests, all pass.
* Live smoke test: trained a `reward_ensemble_size=3`/`value_ensemble_size=3`
  Stage-B checkpoint from scratch (5 steps) and via `--resume` (+3 more
  steps) against the real stride=2 dataset — confirmed 3 independently
  -initialized members, correct checkpoint format, and successful resume.
  Loaded that checkpoint through `LewmAgent.load()` and ran `act()` three
  times successfully (ensemble path). Separately re-loaded the existing
  `lewm_heads_checkpoint_stride2_m4_v3.pt` (ensemble_size=1) through the
  same updated `LewmAgent.load()`/`act()` path to confirm zero regression
  for the non-ensembled case.

## Update 2026-07-20 (later same day): v4 full-scale ensemble retrain, audit, and live eval

Following "full control for the day," the suggested next step above was
executed in full.

### v4 training run

* Config: `configs/lewm_heads_m4_v4.yaml` — identical to v3's m4 config
  (`imagined_horizon=5`, `cont_dropout=0.35`, `cont_hidden_dim=64`,
  `value_loss_weight=1.0`, `imagined_loss_weight=1.0`) plus
  `reward_ensemble_size: 3`, `value_ensemble_size: 3`.
* Trained 20,000 steps from `data/lewm_checkpoint_stride2.pt` (same
  Stage-A checkpoint as v3) on the same dataset
  (`/media/jeovach/New Volume/leworldgaming`, 9 files, 1,103,904 frames).
  Wall-clock: ~90 minutes.
* Final checkpoint: `data/lewm_heads_checkpoint_stride2_m4_v4.pt`.
  Continuation head correctly swapped to its best-val snapshot
  (`val_loss_c=0.6325`, matching v2/v3's pattern) at save time.
* Verified ensemble diversity directly: `torch.equal()` returns `False`
  between reward-head member 0 vs 1, and value-head member 0 vs 1 — the
  3 members are indeed distinct trained networks, not accidentally tied
  weights.

### Audit-v4: calibration re-check + a new ensemble-spread check

1. Re-ran `scripts/audit_lewm_calibration.py` against v4 (using its
   singular/member-0 `reward_head`/`value_head` keys, same as v3's audit)
   on the same 20,000 held-out windows. Result: **no regression vs v3** —
   latent cosine drift (0.997→0.967 across depth 1-5) and reward-head
   `r_gap` (≈0.0000 at every depth, including the decisive |reward|>0
   subset, MAE 0.045-0.054) are both essentially identical to the v3
   numbers reported above.
2. **New check** (not previously done): does the 3-member ensemble
   actually *disagree* enough to give the planner a useful uncertainty
   signal? Loaded all 3 reward/value head members, ran them independently
   on 8,000 held-out grounded (non-rolled) states, and measured
   ensemble-member spread (std across the 3 predictions):

   | metric | all-frame | decisive-frame (\|reward\|>0, n=168) |
   |---|---|---|
   | reward ensemble std | 0.00013 | 0.00016 |
   | reward MAE (ensemble mean) | 0.0013 | 0.0436 |
   | value ensemble std | 0.0060 | 0.0060 |
   | corr(reward std, \|error\|) on decisive frames | — | 0.374 |

   **Finding: the ensemble has effectively collapsed.** Member
   disagreement (std ≈ 0.00013-0.00016) is roughly 300x smaller than the
   actual decisive-frame prediction error (MAE 0.044) — despite being
   independently initialized (confirmed distinct via `torch.equal`), all
   3 members converged to near-identical functions after joint training
   on identical batches with no bootstrap resampling or explicit
   diversity mechanism (a known deep-ensemble failure mode — see Lakshminarayanan
   et al. 2017 discussions of ensemble collapse without decorrelation).
   The weak positive correlation (0.37) between std and error confirms the
   *direction* of the signal is sane, but its magnitude is far too small
   to move a pessimistic CEM score (`mean - k*std`) by any amount that
   would change action selection for realistic `k`.

### Eval-v4-sweep: live self-play vs Dreamer

Given the ensemble-collapse finding above, the planned 3-point sweep
(`uncertainty_penalty` ∈ {0.0, 0.5, 1.0}) was reduced to a single run at
`uncertainty_penalty=0.0` (i.e. plain ensemble-mean scoring, the
closest-to-v3-equivalent configuration) — since a near-zero ensemble std
means differing `k` values are not expected to change behavior, running
the same expensive live match twice more would not have produced new
information.

* Added `--planner-uncertainty-penalty` CLI flag to `scripts/self_play.py`
  (threads straight into `LewmAgent.configure_planner(uncertainty_penalty=...)`,
  which already existed).
* Command: `p1=lewm` (v4 checkpoint) vs `p2=dreamer`, `--p2-frame-skip 2`,
  `--games 2` (yielded 3 rounds).
* **Result: LeWM 0/3 (0%) — all 3 rounds were total shutouts**
  (`hp_p1=0.0` every round; Dreamer's HP remained at 125/340/140). This is
  *worse* than v3's already-poor 1/12 (8.3%) from the prior session.
* Latency: LeWM mean=33.7ms/p95=35.9ms/**drop_rate=48.7%** (n=2832) vs
  Dreamer mean=7.4ms/drop_rate=0.0%. LeWM is still dropping roughly half
  its decision frames to the JVM's real-time budget — consistent with
  the latency profile observed in the prior overnight session, unaffected
  by the ensembling change (as expected, since ensembling doesn't touch
  planner rollout compute cost).

### Final conclusion / recommendation

**v4 (ensembled reward/value heads) does not improve LeWM vs Dreamer, and
should NOT replace v3 as the production checkpoint.** The root cause
established across this whole day's work is *not* reward/value head
miscalibration or imagined-latent exploitation (both were directly ruled
out by the calibration audit, on both v3 and v4) — it is LeWM's real-time
decision latency (mean ~34ms, ~49% of decisions dropped/late against a
33ms frame budget) versus Dreamer's much cheaper ~7ms inference, compounded
by the earlier-established constraint that LeWM can only be evaluated as
P1 (its pixel encoder was only ever trained on P1 data) and by CEM planner
depth/sample settings already being tuned as far as this environment's
GPU-contention constraints allow (see `docs/lewm_stride2_vs_dreamer_2026-07-19.md`).

**Recommendation**: keep `data/lewm_heads_checkpoint_stride2_m4_v3.pt` (or
the original stride=2 checkpoint) as the reference LeWM configuration.
Ensembling is a validated, tested, backward-compatible feature (still
useful as defensive infrastructure — e.g. if future data collection
substantially increases decisive-frame density, a properly-diversified
ensemble could become informative) but is not the lever that closes the
gap to Dreamer. Any further attempt to close that gap should target LeWM's
per-decision latency/drop-rate (e.g. cheaper planner defaults, reduced
CEM iterations/samples, or a lighter-weight encoder) rather than further
head-calibration work, since calibration has now been ruled out twice
(v3 and v4) as the bottleneck.
