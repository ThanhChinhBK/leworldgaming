# Research Notes — 2026-07-16: World-Model Non-Stationarity vs. MctsAiZoning

## 1. Problem statement

The trained LeWM (stride-5, continuation-head) checkpoint still loses a lot
against `MctsAiZoning`. Hypothesis raised: our head training follows the
original single-agent world-model recipe (predict `s' | s, a_own`), which
assumes a roughly stationary environment. In a 2-player fighting game the
"environment" includes an opponent that is itself acting/adapting every
frame, so this assumption may be systematically wrong — especially against a
strong search-based bot whose behavior differs a lot from the scripted/self-play
policies used during most of data collection.

## 2. What the literature says (web research, 2026-07-16)

Standard world-model architectures (Dreamer, JEPA/Genie-style, V-JEPA) are
built around a single-agent assumption: transitions are a stationary function
of the agent's own action, `p(s' | s, a_own)`. Several lines of published work
address exactly the gap we're seeing when a second, adapting agent is part of
the environment:

- **MAMBA** — Egorov & Shpilman, *"Scalable Multi-Agent Model-Based
  Reinforcement Learning"*, arXiv:2205.15023 (2022). Learns a **joint
  action-conditioned world model**: the transition model is conditioned on
  all agents' actions, not just the ego agent's, then used for imagined
  rollouts (CTDE-style).
- **Opponent / self-other modeling** — He et al., *"Opponent Modeling in Deep
  Reinforcement Learning"* (2016); Rabinowitz et al., *ToMnet* (2018,
  arXiv:1802.07740). Train an explicit opponent model (or a "theory of mind"
  latent) that predicts the opponent's action/intent, and condition the
  policy or dynamics model on that prediction.
- **CoDreamer** (2024) and **MACD** (AAMAS 2024) — extend Dreamer-style RSSMs
  to multi-agent settings, explicitly calling out non-stationarity as the
  central obstacle and addressing it via communication-conditioned latents /
  counterfactual imagination.

Common thread across all of them: **condition the dynamics model on (or
predict) the opponent's action**, rather than treating the opponent purely as
unmodeled environment noise. A secondary, complementary thread from
league/self-play literature (AlphaStar, PSRO) is **opponent diversity**:
single-opponent training data generalizes poorly to an unseen, adaptive
opponent, independent of any architecture change.

## 3. Codebase findings

- `pyftg`'s `CharacterData` exposes an `.action` field for **both** players
  every frame (`src/leworldgaming/env/state_vector.py` uses
  `frame_data.get_character(player_number)` / `get_character(not
  player_number)`). This means the opponent's realized action is not truly
  hidden information in FightingICE — it's just never captured today.
- `RecordingAI` (`src/leworldgaming/env/recording_ai.py`) only writes the
  **own** executed action (`action.to_int()`) to the top-level `action`
  column of the replay buffer. The opponent's action is not recorded
  anywhere in `obs/opp/*` (`src/leworldgaming/data/replay_buffer.py`,
  `_PER_CHAR_SCHEMA`).
- The LeWM AR predictor (`src/leworldgaming/agents/lewm/predictor.py`,
  AdaLN-zero conditioning) is conditioned only on the own-action embedding
  from `train_lewm.py` — there is currently no path for opponent-action
  conditioning even if the data existed.
- Current training data (`data/replay.h5`) is small (~4.9k frames in the
  active file) and, per data-collection notes, only two JVM opponents exist
  (`MctsAi23i`, `MctsAiZoning`) alongside scripted self-play policies
  (random/aggressive/defensive). There is no per-frame opponent-identity tag
  stored, so today there's no way to check how much of the training set is
  actually adaptive-opponent (MctsAiZoning-like) data versus scripted
  self-play.

## 4. Candidate improvements (not yet implemented)

Ranked roughly by effort:

1. **Opponent-action conditioning** — record `obs/opp/action` at collection
   time (trivial: `char.action.to_int()` is already available), then extend
   the AR predictor to condition on both `a_own` and `a_opp` (matches
   MAMBA / opponent-modeling literature directly). Requires a schema
   change + a **full Stage-A + Stage-B retrain** to take effect, since the
   predictor's conditioning input changes shape/semantics.
2. **Opponent diversity in training data** — collect (or rebalance) more
   episodes against `MctsAiZoning` specifically (and any other strong bots
   available) so the world model has seen enough of that opponent's dynamics
   distribution. Cheaper than (1) in terms of code, but still requires new,
   possibly long, JVM-based data collection runs.
3. **Explicit opponent-intent auxiliary head** — predict `a_opp` from
   history as an auxiliary task (ToMnet-style), and feed the *predicted*
   distribution into the dynamics model at rollout time (when the true
   future opponent action isn't known). More complex than (1); useful mainly
   if (1) alone doesn't close the gap because open-loop imagination still
   needs an opponent-action prior.

## 5. Attempted implementation (reverted)

Started implementing (1) as a schema-only, additive change:

- Added `"action"` to `_encode_character_dict` / `_empty_char_dict`
  (`state_vector.py`) and to `_PER_CHAR_SCHEMA` (`replay_buffer.py`), so
  both `obs/own/action` and `obs/opp/action` would be recorded per frame.
- Added `ReplayBuffer._backfill_missing_schema_fields()` so older `.h5`
  files would auto-migrate (zero-filled) instead of breaking.
- Verified: `tests/test_lewm_contracts.py` (11/11 pass), a manual
  round-trip smoke test, and confirmed `dreamer_export.py`'s generic
  `GROUPS`-schema reader still worked against a migrated file.

**Decision: reverted.** This is the right direction per the research above,
but wiring it all the way through (predictor conditioning + a full
Stage-A/Stage-B retrain + new data collection) is a large, multi-session
effort. Before committing to it, we want to first benchmark where the 3
already-trained models currently stand.

## 6. Next step (in progress)

Evaluate the **3 existing trained models** head-to-head / vs. built-in bots
first, to establish a baseline and see whether the loss pattern vs.
`MctsAiZoning` is specific to LeWM or shared across all three (which would
suggest a data/opponent-diversity issue rather than a LeWM-specific
architecture gap):

- **LeWM** (stride-5, `data/lewm_checkpoint_stride5.pt` +
  `data/lewm_heads_checkpoint_stride5.pt`)
- **PETS** (`src/leworldgaming/agents/pets`)
- **DreamerV3** (`src/leworldgaming/agents/dreamer`)

Driver: `scripts/evaluate.py` → `leworldgaming.eval.tournament.run_tournament`.

Once baseline win rates vs. `MctsAi23i` / `MctsAiZoning` are known for all
three, revisit this doc to decide whether to invest in opponent-action
conditioning (§4.1), data rebalancing (§4.2), both, or something else
entirely.
