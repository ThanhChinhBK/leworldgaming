# LeWM planner — overnight literature research toward beating Dreamer (2026-07-22)

Continuation of `docs/lewm_mcts_vs_cem_2026-07-21.md` (MCTS lost to CEM live) and
`docs/lewm_calibration_audit_and_ensembling_2026-07-20.md` (the real bottleneck is
reward/value-head *imprecision on rare decisive frames* + a *collapsed* ensemble, NOT
latent drift or head miscalibration in the averaged sense). This doc is a web-literature
pass — no new live matches — to identify the *most promising planner/search direction* given
those two established facts, and to convert vague prior "try MPPI / try a policy prior"
bullets into concrete, evidence-backed, ranked next steps.

## The single most important finding

**The strongest published agent on this exact platform (FightingICE / DareFightingICE) that
is *not* a hand-coded MCTS-with-perfect-simulator bot is RHEAPI** — Rolling Horizon Evolution
Algorithm + an **online-learned opponent model** (Tang et al., "Enhanced Rolling Horizon
Evolution Algorithm with Opponent Model Learning", IEEE T-Games 2020, arXiv:2003.13949). It
took **2nd place in the 2019 competition** (score 122 vs champion ReiwaThunder's 133),
beating every MCTS bot from 2018 by large margins, **while using far less domain knowledge
than the winner** and being the *only* top-5 bot not based on MCTS.

This is directly relevant because RHEAPI is a **population-based forward-planner over discrete
actions with a short horizon** — i.e. structurally *the same family* as our discrete-CEM
planner (`cem_shooting`), not the tree-search family we already showed loses. The paper's
own ablations tell us **where the wins actually came from**, and it is not the search
algorithm:

### RHEAPI's exact configuration (all from the paper, Table I + text)

| Component | Value | Note |
|---|---|---|
| Population size `n` | **7** | *tiny* — comparable to our CEM `num_samples` being small |
| Elites `k` | 1 | |
| Action-sequence length `l` | **4** | our stride-5 horizon is 8 blocks; comparable order |
| Mutation prob `p_m` | **0.85** | *very high* — heavy per-decision exploration |
| Diversity weight `λ` | 0.5 | fitness = `(1−λ)·score + λ·diversity` |
| Shift buffer (warm start) | **tried, dropped** | "does not make much difference … suitable for long-horizon planning and the length here is relatively small" |
| Forward model | game's built-in simulator | (we replace this with LeWM's latent predictor) |
| Fitness score | HP-diff `(hp_self − hp_opp)/max_hp`, ±1 terminal | ~identical to our reward |

**Key ablation result (their Table II/III):** vanilla RHEA (no opponent model) already beats
2018 MCTS bots, but adding the online opponent model lifts mean win-rate by **~10–30 points**
depending on character (e.g. ZEN mean 73.4% → 87.3% with policy-gradient opponent model), and
the *policy-gradient* opponent model beats supervised and Q-learning variants. Crucially, the
opponent model **"provides no advantage in the first round, but usually leads to a
significant improvement in subsequent rounds"** — it is learned live, from scratch, within the
tournament.

### Why this reframes our whole effort

Our `docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md` already flagged, as its final
bullet, the "structural gap not fixable by planner alone: our planner has no opponent model
(plans as if Dreamer's next action is fixed)". The RHEAPI paper is **direct, same-platform,
peer-reviewed evidence that this is the highest-value missing piece** — bigger than any
search-algorithm swap (CEM↔MCTS↔RHEA) or head-recalibration work:

- We spent two prior sessions confirming search-algorithm and head-calibration changes
  *don't* move the needle (MCTS lost; ensembling collapsed; heavier CEM regressed).
- RHEAPI's ablations show, on our exact game, that swapping *the search family itself*
  (RHEA vs MCTS) is a *second-order* effect, while adding an opponent model is a
  *first-order* one. Our planner currently rolls out LeWM's predictor conditioned on
  **our** action only; the predictor was trained on frames where the opponent was doing
  *something*, so at inference it implicitly assumes an "average/marginal" opponent and
  cannot anticipate Dreamer's specific reactive behavior — exactly the deficiency RHEAPI's
  opponent model fixes.

## Recommended direction #1 (highest expected value): add an opponent model to CEM rollouts

Keep the discrete-CEM planner (validated best-so-far). Add a small **opponent action
predictor** and condition the latent rollout on *both* players' actions instead of ours
alone. This is the LeWM analogue of RHEAPI's single biggest win.

Concretely, matching RHEAPI's recipe as closely as our architecture allows:

1. **Model:** a tiny MLP `o_t = OM(state_t)` over the ~40 commandable actions. RHEAPI found
   a *single linear layer* (18 features → 56 softmax), **no hidden layer**, beat MLP/LSTM —
   so this is cheap enough to evaluate inside every CEM rollout step. Our analogue: feed the
   predictor's current latent `z_t` (or the grounded state-vector obs we already have from
   `state_vector.py`) into `OM`, get an opponent-action distribution, and either (a) sample
   the opponent's action per rollout step (RHEAPI's `FORWARD` does exactly this — one-step
   look-ahead opponent inference inside the rollout), or (b) marginalize.
2. **Rollout conditioning:** LeWM's `action_encoder`/`predictor` currently encode only our
   action block. To condition on the opponent we'd need the predictor to accept an opponent
   action too. Two options, in increasing cost:
   - **Cheap, no retrain:** the opponent affects us only through observed reward/next-state;
     since our reward head is `reward_head(z, a_emb)` with our action, the simplest injection
     is to use `OM` to *reweight/penalize* trajectories that assume an unrealistically
     passive opponent — i.e. a pessimistic HP-diff correction term. Weakest form.
   - **Proper, needs Stage-B retrain:** add an opponent-action input channel to the
     action encoder and re-train Stage-B heads on replay data that *has* the recorded
     opponent action per frame (FightingICE `FrameData` exposes both characters — check
     whether `collect_data.py`/`_replay_utils.py` already stores opponent actions; if so
     this is mostly a data-plumbing + head-retrain job, not new data collection).
3. **Online training (the part that matters):** train `OM` **live, at round boundaries**, on
   the observed `(state, opponent_action)` pairs from the round just played, exactly as
   RHEAPI does ("opponent state-action pairs recorded in a dataset … trained by the latest
   dataset at the end of the round … ~5 s between rounds"). This is what let RHEAPI adapt to
   specific opponents. Our self-play harness plays Dreamer repeatedly, so an online-adapting
   `OM` would specifically learn Dreamer's tendencies across rounds — the same
   round-2-onwards improvement curve RHEAPI reported. Policy-gradient-trained `OM` beat
   supervised in their study (it weights *impactful* rare actions like deadly specials by
   reward instead of frequency — precisely our "rare decisive frames" problem from the
   calibration audit), but supervised cross-entropy is far simpler and was a close second;
   **start with supervised, it's a 1-layer softmax.**

Expected payoff: RHEAPI's own numbers suggest 10–30 win-rate points on this platform from
the opponent model alone. Even a fraction of that would move us past Dreamer from the current
~33%.

## Recommended direction #2 (cheap, no retrain): RHEA-style diversity term + high mutation

Independent of the opponent model, RHEAPI's fitness explicitly *adds a diversity bonus*
(`λ=0.5`, `f_div` = 1 − mean per-gene occurrence frequency across the population) and uses a
*very high* mutation rate (`p_m=0.85`). Both fight premature convergence to a single
repeated action — which is **exactly the failure mode our own `cem_shooting` docstring
(point 3) and the stride5 doc's "stuck holding one action for 200+ decisions" observation
describe.** Our CEM already has `min_prob` (uniform floor) and sampled-not-argmax execution
as partial mitigations, but not an *explicit population-diversity reward*. Adding an `f_div`
term to `_score_action_sequences` (penalize candidates whose action multiset is over-
represented among the current sample batch) is a ~15-line, no-retrain change that mirrors
the champion-adjacent bot's design and directly targets our observed action-lock-in.

## Recommended direction #3 (planner robustness to the collapsed ensemble): MPPI soft update + TS∞-style per-rollout head sampling

Our calibration audit found the reward/value ensemble **collapsed** (member std ≈300× smaller
than actual decisive-frame error), so `mean − k·std` pessimism (`uncertainty_penalty`) is
inert. Two literature-backed fixes:

- **Diversify the ensemble properly.** Uncertainty-Guided CEM (Rafailov/Yu-style, arXiv:2111.04972)
  and PETS get non-collapsed ensembles via **bootstrap resampling of training data per member**
  and **TS∞ trajectory sampling** (each rollout *particle* commits to *one* fixed ensemble
  member for the whole horizon, instead of averaging members every step). Our Stage-B trainer
  currently trains all members on *identical* batches (the documented collapse cause). Adding
  per-member bootstrap masks + TS∞-style per-rollout member selection is the standard,
  published recipe to make the disagreement signal real — a prerequisite for *any*
  uncertainty-aware planning (pessimistic *or* exploratory) to do anything at all here.
  Their penalty form is `R_i = Σ r(s_t,a_t) − β·ω_i` with `ω_i` the horizon-averaged,
  per-step-normalized ensemble variance — directly portable to `_score_action_sequences`
  once the ensemble actually disagrees.
- **MPPI soft (Boltzmann) elite update instead of CEM's hard top-k** (Williams et al.,
  Information-Theoretic MPC). Already flagged in the prior doc as a cheap upgrade; the
  literature detail worth adding is the exact reweighting: instead of averaging the top-k
  elites uniformly, weight *every* sampled sequence by `w_i ∝ exp(−(1/λ)·cost_i)` and refit
  the per-timestep categorical distribution as the softmax-weighted action histogram. On a
  noisy/sparse score function (our decisive-frame situation) this is more robust than a hard
  cutoff because a single over-optimistic candidate can't fully capture an elite slot — its
  influence is bounded by its exponential weight. ~10 lines in `cem_shooting`'s refit step,
  no retrain.

## Explicitly de-prioritized (evidence against, this session)

- **MCTS / MuZero-style tree search:** already lost live (`docs/lewm_mcts_vs_cem_2026-07-21.md`),
  and RHEAPI's paper independently shows RHEA (population) matches/beats MCTS on this platform
  at equal real-time budget — consistent with our result. Only worth revisiting *with* a
  trained policy prior (Sampled MuZero, Hubert et al. ICML 2021), which is a bigger lift than
  directions #1–#3 and addresses a weakness (no prior) that RHEA/CEM don't even have.
- **Gradient-based / Gumbel-softmax planning:** our discrete, multi-modal, noisy-score
  setting is the worst case for single-trajectory gradient descent (mode collapse + more
  aggressive exploitation of the imperfect heads than sampling). Literature (shooting-vs-
  gradient surveys) agrees shooting wins in low-dim discrete/multimodal regimes.
- **Pure heavier CEM search:** already shown to *regress* past a sweet spot
  (`docs/lewm_stride5_reretrain_vs_chunking_2026-07-20.md`, Follow-up 3) via model
  exploitation — more search into imperfect heads is actively harmful.

## Concrete ranked next actions

1. **[Biggest expected win] Opponent model conditioning** (direction #1). **Data-plumbing
   status verified this session:** the opponent's discrete action *is* available live and in
   replay via `FrameData.get_character(not player_number).action` (confirmed
   `pyftg`'s `CharacterData` exposes `.action`), and the state vector
   (`env/state_vector.py`) already encodes rich opponent *state* (position, speed, attack
   startup/active/type, control flags) — but **`RecordingAI._process_frame` currently logs
   only our own action** (`env/recording_ai.py`, `self._buffer.add(action=action.to_int())`),
   not the opponent's. So:
   - **Immediate (no new collection needed for an online `OM`):** the online round-boundary
     `OM` RHEAPI uses does **not** need historical replay at all — it trains live on
     `(opponent_state, opponent_action)` pairs observed during the current match, both of
     which are already in `FrameData` at inference time. This can be built purely in
     `LewmAgent`/`self_play.py` with zero dataset changes.
   - **For rollout conditioning (predictor sees opponent action):** requires adding one line
     to `RecordingAI` to also log `opp.action.to_int()`, then re-collecting (or, if any
     existing `.h5` retains raw `FrameData`, back-filling) + a Stage-B retrain with the
     opponent-action input channel.
   - **Recommended order:** build the live-trained `OM` first (cheapest, matches RHEAPI's
     single biggest ablation win, no data/retrain), measure it, then decide if the heavier
     predictor-conditioning retrain is worth it.
2. **[Cheap, same day] RHEA diversity term + higher mutation/`min_prob`** in `cem_shooting`
   (direction #2) — direct anti-lock-in, mirrors the champion-adjacent bot.
3. **[Cheap, same day] MPPI soft-weighted refit** (direction #3b) — robustness to noisy
   decisive-frame scores.
4. **[Medium] Properly de-correlated ensemble** (per-member data bootstrap + TS∞ rollout
   sampling) (direction #3a) — makes uncertainty-aware scoring functional, unblocking both
   pessimism and (sign-flipped) exploration bonuses.

Items 2 and 3 are near-zero-risk, no-retrain, and independently testable in a single live
self-play batch each; item 1 is the one the same-platform literature most strongly predicts
will actually beat Dreamer.

## Sources

- Tang, Shao, Zhao, Zhu et al., "Enhanced Rolling Horizon Evolution Algorithm with Opponent
  Model Learning" (RHEAPI), IEEE Transactions on Games 2020 — arXiv:2003.13949. Same platform
  (FightingICE), 2nd place 2019 competition, opponent-model ablations.
- Self-Adaptive RHEA, IEEE CoG 2020 — online adaptation of RHEA search-control params.
- "Risk Sensitive MBRL using Uncertainty Guided Planning" (Uncertainty-Guided CEM) —
  arXiv:2111.04972. Bootstrap-ensemble variance penalty in CEM; TS∞ particle propagation.
- Chua et al., PETS, NeurIPS 2018 — probabilistic ensembles + trajectory sampling (TS1/TS∞).
- Pinneri et al., iCEM (colored-noise, memory), CoRL 2021 — our current planner's family.
- Williams et al., Information-Theoretic MPC (MPPI) — soft/Boltzmann exponential-weighted
  update vs hard top-k.
- Hubert et al., Sampled MuZero, ICML 2021 — MCTS in large/continuous action spaces via a
  policy prior (only relevant if MCTS is revisited).
