# PETS CEM config audit vs. original paper (2026-07-24)

Per user request, before finalizing the PETS training/eval this session,
web-searched and fetched the official reference implementation
(`kchua/handful-of-trials`, Chua et al. 2018, arXiv:1805.12114) to check our
`configs/pets.yaml`/`pets_vsdreamer.yaml` + `cem_planner.py` against the
paper's actual defaults, rather than assuming the repo's existing numbers
were already faithful.

## Official reference values (per-environment `dmbrl/config/<env>.py`)

| | cartpole | halfcheetah | pusher |
|---|---|---|---|
| PLAN_HOR | 25 | 30 | 25 |
| CEM popsize | 400 | 500 | 500 |
| CEM num_elites | 40 | 50 | 50 |
| CEM max_iters | 5 | 5 | 5 |
| CEM alpha | 0.1 | 0.1 | 0.1 |
| ensemble size | 5 | 5 | 5 |
| hidden units | 500 | 200 | 200 |
| hidden layers | 3 | 3 | 3 |
| npart (particles/candidate) | 20 | 20 | 20 |
| propagation mode | TSinf (best-performing variant per paper) | | |

(`default.py`'s `create_config`/`_create_ctrl_config` confirms `opt-type`
defaults to `"CEM"` and `prop-type` defaults to `"TSinf"` in the CLI/config
plumbing.)

## Our config (`configs/pets.yaml`, `pets_vsdreamer.yaml`)

| | ours |
|---|---|
| horizon (PLAN_HOR) | 15 |
| num_candidates (popsize) | 200 |
| num_elites | 20 |
| num_iters (max_iters) | 4 |
| alpha | **not implemented** |
| ensemble_size | 5 |
| hidden | 200 |
| num_layers | 3 |
| npart | **effectively 1** (see below) |
| propagation mode | **TS1**, per `dynamics.py`'s own docstring |

## Findings

1. **popsize/num_elites are both ~2-2.5x smaller than every official env**
   (200/20 vs 400-500/40-50), but the **elite fraction is preserved exactly**
   at 10% in both (20/200 = 40/400 = 50/500 = 0.1). Likely a deliberate,
   reasonable scale-down given this project's much tighter real-time
   decision budget (fighting-game MPC replanning every ~16-133ms depending
   on frame_skip) vs. the original's offline/non-real-time MuJoCo MPC loop.
2. **plan_hor=15 vs 25-30 in every official env** — again smaller, plausibly
   an intentional real-time-budget adaptation (each extra horizon step is
   another full ensemble forward pass per candidate).
3. **max_iters=4 vs 5** — close, minor.
4. **`alpha` (CEM mean-smoothing momentum) is not implemented at all.**
   Official CEM refits the sampling distribution each iteration as
   `dist = alpha * old_dist + (1-alpha) * new_dist` (continuous-Gaussian
   CEM's momentum term, preventing the distribution from collapsing too
   aggressively iteration-to-iteration). Our discrete `cem_planner.py`
   does a **hard replace** every iteration (`logits = new_logits`, no
   blending with the previous iteration's `logits`). This is a **real,
   unintentional deviation** — it removes a core CEM stability mechanism
   present in every official env config (`alpha: 0.1` in all three).
   Note: this is a *different* codepath from `agents/lewm/planner.py`'s
   `cem_shooting`, which *does* implement momentum (`momentum` param,
   already tested/tuned this session) — the PETS-specific
   `cem_planner.py` was simply never given the same treatment.
5. **Propagation mode is TS1, not the paper's actual best-performing
   default (TSinf).** `dynamics.py`'s `predict()` docstring explicitly
   labels itself "TS1-style next-state prediction" — each of the 200 (or
   400-500 in the original) candidates is assigned **one fixed ensemble
   member for its entire rollout** (`members = torch.randint(...)` sampled
   once per candidate, reused every horizon step). The paper's own ablation
   table (and every official env's CLI default, `prop-type` defaulting to
   `"TSinf"`) uses **TSinf**: each of the `npart` particles per candidate
   independently resamples which ensemble member it uses **at every
   timestep**, averaging over epistemic *and* aleatoric uncertainty more
   thoroughly per candidate. TS1 is presented in the paper as one of
   several ablated/weaker propagation variants, not the default.
6. **npart (particles per candidate) is effectively 1, not 20.** The
   official implementation scores each candidate action sequence by
   averaging its cost over `npart=20` independent stochastic rollouts
   (particles) through the ensemble, reducing variance in each candidate's
   score estimate. Our `cem_planner.py` runs exactly one rollout per
   candidate (one `member_ids` draw, one `predict()` per horizon step per
   candidate) — there is no inner particle-averaging loop at all. Combined
   with finding 5, this means our per-candidate score is a single noisy
   sample from one ensemble member's stochastic dynamics, rather than the
   paper's more robust multi-particle, multi-member average.

## Assessment: which deviations matter and which don't

- **Findings 1-3 (smaller popsize/num_elites/plan_hor, same elite
  fraction, similar max_iters)**: reasonable, likely intentional real-time
  adaptations. Not recommended to change without also relaxing this
  project's frame_skip/latency constraints (already found to be a binding
  constraint for PETS this session — see `docs/pets_final_training_2026-07-24.md`,
  ~100ms/decision at these settings already forces `--p1-frame-skip 8`).
  Increasing popsize/plan_hor further would only worsen PETS's already
  serious real-time deficit.
- **Findings 4-6 (no alpha momentum, TS1 instead of TSinf, no particle
  averaging) are genuine, unintentional deviations from the paper**, not
  deliberate real-time adaptations — they don't cost extra planning time
  in the same way (alpha is free; TSinf/npart-averaging do cost more
  ensemble forward passes per candidate, proportional to npart, but this
  is a core part of what makes PETS's uncertainty-aware planning actually
  work in the paper — arguably the single most distinctive claim of PETS
  as an algorithm, per its own title/abstract: "trajectory sampling").
  Given PETS already loses 0/9 live and is confirmed non-competitive with
  LeWM (see `docs/pets_final_training_2026-07-24.md`), and any of these
  fixes would add compute cost PETS can already barely afford in real time,
  **no code changes are recommended purely to "match the paper" here** —
  PETS's role in this project is a baseline comparison, and the existing
  result (0/9) already conclusively answers the planner-quality question
  this session set out to answer (LeWM beats PETS decisively). Faithfully
  reproducing TSinf/npart/alpha would only be worth the added latency cost
  if PETS were being considered as a serious promotion candidate, which it
  is not.

## Conclusion

The PETS CEM config is **partially matched, partially deviated** from the
original paper:
- Sample-budget hyperparameters (popsize, num_elites, plan_hor, max_iters)
  are smaller but proportionally consistent (same 10% elite fraction),
  and are reasonable real-time-budget adaptations for this project.
- The propagation method (TS1 vs. paper's TSinf) and the missing CEM alpha
  momentum term are genuine implementation gaps versus the paper's actual
  best-performing configuration — flagging honestly here per the user's
  request, but **not fixing them**, since (a) TSinf/npart-averaging would
  make PETS's already-binding real-time budget problem worse, not better,
  and (b) PETS has already been conclusively shown non-competitive with
  LeWM in this project's live evaluation, so further paper-fidelity work
  on PETS is not a good use of time relative to the project's actual goal
  (finding a planner that beats DreamerV3).
