# leworldgaming

Research project comparing **LeWorldModel (JEPA)**, **DreamerV3**, and **PETS** controllers in **DareFightingICE**. Plan & motivation: [`docs/gemini_research.md`](docs/gemini_research.md).

## Research report and results

**Which World Model Is the Best Fighter? LeWM vs. DreamerV3 vs. PETS**

Chinh Nguyen-Thanh, 2026.

Read the [paper (PDF)](paper/main.pdf) or its [LaTeX source](paper/main.tex).
This is a self-published, non-peer-reviewed research report hosted in this
repository, not an arXiv publication; no arXiv identifier or DOI is assigned here.

The report compares fixed, offline-trained controllers: pixel-based LeWM
with CEM planning, state-vector DreamerV3 with a learned actor, and a
state-vector PETS-style ensemble with CEM planning. Across **135 reported
lockstep rounds**, the outcomes were:

| Matchup (first controller's perspective) | Match 1 | Match 2 | Match 3 | Total wins-losses | Win rate |
|---|---|---|---|---|---|
| DreamerV3 vs. LeWM | 11-4 | 14-1 | 7-8 | 32-13 | 71.1% |
| LeWM vs. PETS | 9-6 | 8-7 | 11-4 | 28-17 | 62.2% |
| DreamerV3 vs. PETS | 15-0 | 15-0 | 15-0 | 45-0 | 100.0% |

Each match contains 15 rounds. The report uses the first three completed
matches per pair in scheduled order (45 rounds per pair), with no draws.
The early-stopped run saved 150 completed rounds; the report excludes one
surplus 15-0 DreamerV3/PETS match to give each pair equal coverage, as well
as all interrupted matches. The reported subset is not a completed full
tournament schedule.

| Controller | Mean decision latency | Decisions over the 16.67 ms budget |
|---|---|---|
| DreamerV3 | 9.3 ms | 0.7% |
| LeWM | 70.3 ms | 100% |
| PETS | 126.9 ms | 100% |

Latencies are weighted by decision count over the nine reported matches,
measured on an NVIDIA GeForce RTX 5060 Ti (16 GB) and AMD Ryzen 5 3600.
The JVM waited for both controllers on every raw frame (`--input-sync`):
these are **lockstep control outcomes, not real-time win rates**.

**Takeaway and limits.** DreamerV3 had the strongest combination of round
outcomes and decision speed for these checkpoints, with LeWM next in round
outcomes. This does **not** establish general architectural superiority:
inputs, training data, and compute budgets differ; LeWM always played P1;
and no comparison against the platform's standard search-based agent was
performed. Only one checkpoint per method was evaluated, rounds are
correlated, and the DreamerV3/LeWM per-match spread is substantial.
Exact spectator-image synchronization was not verified. The planned
100,000-step training extensions were not completed.

The PDF, source, figures, and archived Dreamer training-curve CSV are
included in `paper/`. Raw match logs, replay datasets, and model checkpoints
remain local under ignored data paths and are not distributed with this
repository. See [running the benchmark](#running-the-head-to-head-benchmark)
for checkpoint requirements and protocol details.

## Citation

If you use this research, please cite the repository-hosted report:

> Nguyen-Thanh, Chinh. (2026). *Which World Model Is the Best Fighter? LeWM vs. DreamerV3 vs. PETS*. Unpublished research report. https://github.com/ThanhChinhBK/leworldgaming

```bibtex
@misc{nguyenthanh2026bestfighter,
  author       = {Nguyen-Thanh, Chinh},
  title        = {Which World Model Is the Best Fighter? {LeWM} vs. {DreamerV3} vs. {PETS}},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Unpublished research report; not peer reviewed},
  url          = {https://github.com/ThanhChinhBK/leworldgaming}
}
```

For reproducibility, also record the commit hash of the version you used.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync --extra dev
```

Picks the right torch wheel automatically (CPU/MPS on macOS, CUDA on Linux).

## Run the game

**On Mac (native — full window, real sprites):**

```bash
brew install openjdk@21
echo 'export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

make fetch-native     # one-time, ~50MB into vendor/
make game-native      # AI vs AI mode (for collection)
make game-play        # human keyboard play
```

**On Linux training box (native — fastest for pixel collection):**

```bash
sudo apt install openjdk-21-jdk
make fetch-native           # one-time download
make game-native-linux      # GPU-accelerated rendering, ~10× faster than Docker
```

**On Linux training box (docker — alternative):**

```bash
make game             # MODE=fast (no rendering)
make game-pixels      # MODE=pixels (LeWM data, slower — software rendering via Xvfb)
make game-watch       # MODE=watch (VNC at vnc://localhost:5900, password: watch)
make game-stop
```

See [`docker/fightingice/README.md`](docker/fightingice/README.md) for docker details.

## Collect data

With the game running in another terminal:

```bash
uv run python scripts/collect_data.py --games 1                            # state vectors only
uv run python scripts/collect_data.py --games 1 --pixels                   # + 224×224 RGB frames
uv run python scripts/collect_data.py --games 1 --pixels --image-size 84   # smaller pixels (faster on Mac)
```

Default output: `/media/jeovach/New Volume/leworldgaming/replay.h5` (override with `--out`).

### Large-scale pixel collection (200 games, diverse policies)

Start the game with `make game-native-linux`, then run batches:

```bash
export DATA_DIR="/media/jeovach/New Volume/leworldgaming"

uv run python scripts/collect_data.py --games 20 --policy-p1 random --policy-p2 random --seed 1 --pixels --out "$DATA_DIR/01_random_v_random.h5"
uv run python scripts/collect_data.py --games 20 --policy-p1 aggressive --policy-p2 random --seed 2 --pixels --out "$DATA_DIR/02_aggressive_v_random.h5"
uv run python scripts/collect_data.py --games 20 --policy-p1 defensive --policy-p2 random --seed 3 --pixels --out "$DATA_DIR/03_defensive_v_random.h5"
uv run python scripts/collect_data.py --games 20 --policy-p1 mixed --policy-p2 random --seed 4 --pixels --out "$DATA_DIR/04_mixed_v_random.h5"
uv run python scripts/collect_data.py --games 20 --policy-p1 random --policy-p2 aggressive --seed 5 --pixels --out "$DATA_DIR/05_random_v_aggressive.h5"
uv run python scripts/collect_data.py --games 20 --policy-p1 aggressive --policy-p2 defensive --seed 6 --pixels --out "$DATA_DIR/06_aggressive_v_defensive.h5"
uv run python scripts/collect_data.py --games 20 --policy-p1 random --policy-p2 MctsAi23i --seed 7 --pixels --out "$DATA_DIR/07_random_v_mcts23i.h5"
uv run python scripts/collect_data.py --games 20 --policy-p1 aggressive --policy-p2 MctsAi23i --seed 8 --pixels --out "$DATA_DIR/08_aggressive_v_mcts23i.h5"
uv run python scripts/collect_data.py --games 20 --policy-p1 mixed --policy-p2 MctsAiZoning --seed 9 --pixels --out "$DATA_DIR/09_mixed_v_mctszoning.h5"
uv run python scripts/collect_data.py --games 20 --policy-p1 defensive --policy-p2 MctsAiZoning --seed 10 --pixels --out "$DATA_DIR/10_defensive_v_mctszoning.h5"
```

Restart the game engine between batches: `make game-stop && make game-native-linux`.

### Compress after collection

Pixel data is written uncompressed for speed during collection (~32GB per batch). Compress afterwards to save disk:

```bash
uv run python scripts/compress_replay.py --all    # compress all .h5 in DATA_DIR (3-5× reduction)
uv run python scripts/compress_replay.py "$DATA_DIR/01_random_v_random.h5"   # single file
```

Output: HDF5 files with raw primitives stored in named groups so a single collection feeds all three trainers (LeWM / Dreamer / PETS) via per-method dataloader views in `src/leworldgaming/data/views.py`:

```
obs/own/{hp,energy,x,y,speed_x,speed_y,state,front,control,
         remaining_frame,hit_confirm,
         atk_is_live,atk_start_up,atk_active,atk_hit_damage,atk_type}
obs/opp/{...mirrored...}
obs/global/{current_round,current_frame,proj_self,proj_opp,max_hp,max_energy}
action  reward  done  is_first  cont  episode_starts
state_vector (N, 52) float32         — legacy flat mirror; LeWM Stage-B probe target
pixels (only when --pixels is set; uint8, CHW)
```

`is_first` flags episode boundaries (Dreamer requires this to reset the RSSM hidden state); `cont` is `1 − done`. `state_vector` is the legacy 52-dim flat form (`obs_dict_to_legacy_vector`) — written alongside the named groups so the LeWM Stage-B head trainer's linear probe and any other tool expecting the legacy layout work without re-derivation.

> **`--pixels` is LeWM-only.** Dreamer (proprio mode) and PETS train on the same HDF5 without it. Skip the flag for state-vector-only collection runs to halve disk usage.

## Inspect data

```bash
uv run python scripts/extract_replay.py --stride 30   # dumps PNGs + metadata.csv to data/extracted/
open data/extracted/frames                            # browse in Finder
```

## Train LeWM (two stages)

LeWM trains in two stages so the JEPA representation objective stays clean
and isolated from reward-driven heads:

| Stage | What | Output |
|---|---|---|
| **A** — JEPA pretraining | next-embedding prediction + SIGReg, encoder-grounded only | `data/lewm_checkpoint.pt` |
| **B** — head training | freeze JEPA, fit reward / continuation / value / probe heads on the same replay | `data/lewm_heads_checkpoint.pt` |

Stage A is what the original LeWM paper trains. Stage B is what the
benchmark plan (see `docs/benchmark_plan.md`) needs so MCTS can score tree
nodes — without it the latent-space planner has no learned reward / value
signal.

### 1. Collect replay data with pixels

Start the game, then record games with `--pixels` (224×224 RGB frames required by the ViT encoder):

```bash
# Mac (native renderer)
make game-native                    # terminal 1
uv run python scripts/collect_data.py --games 5 --pixels   # terminal 2

# Linux (docker)
make game-pixels                    # terminal 1
uv run python scripts/collect_data.py --games 5 --pixels   # terminal 2
```

### Policy options for `--policy-p1` / `--policy-p2`

| Value | What it is | Where it runs |
|---|---|---|
| `random` | Uniform over the 40 playable actions, sticky for 8 frames (see `env/policies.py`) | Python (pyftg client) |
| `aggressive` | 80% attacks, 20% movement/guard | Python (pyftg client) |
| `defensive` | 70% guard/movement, 30% attacks | Python (pyftg client) |
| `mixed` | Cycles random→aggressive→defensive each game (broadest coverage) | Python (pyftg client) |
| `noop` (alias `neutral`) | Always `Action.NEUTRAL` — passive baseline | Python (pyftg client) |
| `MctsAi23i` | Iteration-capped MCTS, the canonical DareFightingICE 7.x training opponent | JVM (loaded server-side from `vendor/fightingice/data/ai/MctsAi23i.jar`) |
| `MctsAiZoning` | MCTS variant with zoning heuristics | JVM (`MctsAiZoning.jar`) |

When a JVM AI name is passed, `collect_data.py` does **not** spin up a Python agent for that slot — in `--pyftg-mode` the engine resolves the class name against `data/ai/*.jar` and instantiates it in-process. At least one slot must be a Python policy so transitions get recorded.

> Older AIs (`MctsAi`, `KickAI`, `Sandbox`, `Thunder`, `BlindAI`, `ErheaPi`) are **not** bundled in DareFightingICE 7.1 — only the two MCTS jars above are. To use others, drop their compiled `.jar` into `vendor/fightingice/data/ai/` and add the class name to `JVM_AIS` in `scripts/collect_data.py`.

**Examples:**

```bash
# Random P1 vs the canonical MCTS opponent (BlindAI-paper recipe)
uv run python scripts/collect_data.py --games 30 --pixels --policy-p2 MctsAi23i

# Mixed policy (cycles strategies each game) vs MCTS zoning
uv run python scripts/collect_data.py --games 30 --pixels --policy-p1 mixed --policy-p2 MctsAiZoning

# Aggressive vs defensive (varied combat dynamics)
uv run python scripts/collect_data.py --games 30 --pixels --policy-p1 aggressive --policy-p2 defensive

# Self-play random (maximum entropy, no JVM AI involved)
uv run python scripts/collect_data.py --games 30 --pixels

# MCTS-vs-random with --no-record-p2 to halve storage
uv run python scripts/collect_data.py --games 30 --pixels \
    --policy-p1 MctsAi23i --policy-p2 random --no-record-p2
```

This writes `data/replay.h5` containing `pixels (N,3,224,224)`, `action`, `done`, `episode_starts`, etc.

> **Tip:** Mix opponents for diversity. Splitting a 100-game run across `MctsAi23i`, `MctsAiZoning`, and self-play random gives broader state coverage than 100 games of any single matchup.

### 2. Train

```bash
uv run python -m leworldgaming.training.train_lewm --num-steps 1000
```

Or from a script:

```python
from leworldgaming.training.train_lewm import train
results = train(num_steps=1000)
```

Config is loaded from `configs/lewm.yaml` by default. Key knobs:

| Key | Default | What it controls |
|-----|---------|-----------------|
| `encoder_depth` | 12 | ViT depth (6 for fast iteration, 12 for ViT-tiny parity) |
| `history_size` | 3 | AR context window — predictor sees this many past frames |
| `predictor_depth` | 6 | Transformer layers in the AR predictor |
| `batch_size` | 128 | Sequences per step (matches the LeWM source config) |
| `lr` | 5e-5 | AdamW learning rate (matches the LeWM source config) |
| `sigreg_lambda` | 0.09 | Weight of SIGReg anti-collapse regularizer |

Override any key via CLI or kwarg:

```bash
uv run python -m leworldgaming.training.train_lewm --num-steps 2000 --batch-size 8 --encoder-depth 6
```

### 3. What to watch during training

```
step=  29 train pred=0.1676 sigreg=0.9686 |z|=13.29 grad=1.22
step=  29  val  pred=0.2140 sigreg=0.8117 |z|=10.21
```

- **pred** — MSE prediction loss (lower = better next-frame prediction)
- **sigreg** — regularizer loss (pushes embeddings toward N(0,I))
- **|z|** — mean embedding L2 norm; healthy range ≈ √latent_dim ≈ **13–14** for dim=192. Collapse → 0, explosion → >>14.
- **grad** — gradient norm after clipping (should stay ≤ `grad_clip`)

### 4. Checkpoint & inference

Training saves to `data/lewm_checkpoint.pt` (configurable via `ckpt_path`). Load for inference:

```python
from leworldgaming.agents.lewm.agent import LewmAgent

agent = LewmAgent(device="mps")  # or "cuda"
agent.load("data/lewm_checkpoint.pt")  # rebuilds architecture from stored config
action = agent.act({"pixels": frame_tensor})  # (3, 224, 224) float
```

### Architecture

```
ViT-tiny encoder → Projector → AR Predictor → pred_proj
                                      ↑ conditioned on ActionEncoder
```

The AR predictor uses causal self-attention + AdaLN-zero conditioning on per-step action embeddings, so one forward pass yields `T` parallel next-step predictions during training.

### 5. Stage B: head training for latent random-shooting planning

After Stage A converges, fit the reward / continuation / value / probe
heads on the same replay so LeWM exposes `(z, a) → z', r̂, ĉ, V̂`.
JEPA components are frozen; only the heads update. The live planner scores
sampled latent trajectories with trained reward/continuation/value heads;
this repository does not currently implement tree-search MCTS.

```bash
uv run python scripts/train.py --agent lewm --stage b --steps 20000
```

Heads added (each a small MLP on top of the latent `z`):

| Head | Input | Loss | Notes |
|---|---|---|---|
| `RewardHead` | `(z, a_emb)` | twohot CE on HP-delta | TD-MPC2 / DreamerV3 discrete-regression bins |
| `ContinuationHead` | `z` | balanced BCE on `1 - done` | rollout termination signal |
| `ValueHead` | `z` | twohot CE on λ-return | bootstrap via EMA target net |
| `LinearProbe` | `z` | MSE on physical targets | finally trained — used by `planner.py` |

Plus an optional **imagined-rollout consistency** loss: roll the frozen
predictor forward `imagined_horizon` steps and apply the same reward +
continuation losses on the predictor-rolled latents `ẑ_{t+k}`. This is
what makes the heads reliable on the trees MCTS will explore — without
it, heads only ever see encoder-grounded `z`.

Config knobs (`configs/lewm_heads.yaml`):

| Key | Default | Effect |
|---|---|---|
| `reward_loss_weight` | 1.0 | enable / scale reward head |
| `cont_loss_weight` | 1.0 | enable / scale continuation head |
| `value_loss_weight` | 0.0 | set > 0 to enable value head + λ-return |
| `imagined_horizon` | 0 | set > 0 to enable predictor-rolled supervision |
| `imagined_loss_weight` | 0.0 | scale imagined consistency loss |
| `probe_loss_weight` | 0.0 | set > 0 to actually train the linear probe |

Defaults disable everything except reward/continuation so the bring-up is
incremental — start there, add value, then imagined, then probe. The
benchmark plan recommends turning all four on (`value=0.5`,
`imagined_horizon=5`, `imagined=1.0`, `probe=0.1`).

#### What to watch during Stage B

```
step= 100 train r=0.05 c=0.001 v=0.6 r_im=0.06 c_im=0.001 probe=0.4 |z|=27 grad=2.0
```

- `r` / `r_im` — encoder-grounded vs predictor-rolled reward CE; gap = compounding error in the world model
- `v` — λ-return CE; should plateau, not collapse to 0 (bootstrap from EMA target keeps it honest)
- `probe` — MSE on `[hp_diff, hp_self, hp_opp, distance]` extracted from `state_vector`

The Stage-B checkpoint is self-contained: it carries the Stage-A weights
plus the four heads, so `LewmAgent.load("data/lewm_heads_checkpoint.pt")`
gives the latent random-shooting planner access to the trained heads.

## Train DreamerV3 (offline)

DreamerV3 runs in **vector / proprio mode** (per `gemini_research.md` §5) — its RSSM consumes a side-canonicalized 42-d state vector built from the same primitives PETS uses, plus one-hot expansion of the discrete `state` and `atk_type` enums. Pixels are LeWM-only.

The vendored `external/dreamerv3-torch` runs purely from the same HDF5 replay — no live env needed. The trainer exports each episode to a per-episode `.npz` file once (cached in `data/dreamer_episodes/`), then drives `WorldModel._train` + `ImagBehavior._train` directly:

Exports use incoming actions: observation/reward at timestep `t` are paired
with the action from `t-1`, with a zero initial action. Corrected checkpoints
carry `action_alignment: incoming_action_v1`. Legacy caches are rejected
rather than silently reused; preserve them separately and explicitly rebuild
exports from valid original replays before starting fresh training. Legacy
Dreamer checkpoints cannot be resumed as if they had corrected alignment.

```bash
uv run python scripts/collect_data.py --games 5                # produce data/replay.h5
uv run python -m leworldgaming.training.train_dreamer --num-steps 1000
```

Defaults in `configs/dreamer.yaml`: 42-d vector observation (`DREAMER_STATE_DIM`), batch 16×64, model_lr 1e-4, actor/critic_lr 3e-5, imag_horizon 15. Inherits the upstream `dmc_proprio` encoder/decoder regex (`mlp_keys='.*'`, `cnn_keys='$^'`). Online inference is available through `scripts/play.py` and `scripts/self_play.py`, which reconstruct the trained world model and actor from the checkpoint.

**Side symmetry.** State observations are canonicalized at view time so own is always on the left of opp (mirroring x / speed_x / front when needed). A model trained on P1-collected data deploys directly as P2 with no extra changes.

## Train PETS (state-vector ensemble)

Probabilistic-ensemble dynamics over the 26-d continuous state primitives (`PETS_STATE_DIM`), discrete-action variant with categorical CEM and TS-infinity trajectory sampling (one fixed ensemble member per trajectory). The network retains 56 action labels, while inference restricts decisions to commandable actions. Reward is computed analytically from HP primitives — no learned reward head. This is a PETS-style adaptation: members train on a shared batch rather than independent bootstrap datasets.

```bash
uv run python scripts/collect_data.py --games 5               # pixels not required
uv run python -m leworldgaming.training.train_pets --num-steps 1000
```

Defaults in `configs/pets.yaml`: 5-member ensemble, hidden=200, num_layers=3, batch 256, lr 1e-3. Inference-time CEM planner: horizon=15, 200 candidates, 20 elites, 4 iterations — tune these down (`planner_horizon`, `planner_num_candidates`) if you blow the 16.67 ms frame budget; the agent wraps each `act()` call in `FrameBudget`.

**Side symmetry.** Same canonicalization as Dreamer — `obs_dict_to_pets_vector` mirrors x / speed_x / front when own is on the right. Training and inference both consume the canonical view, so P1-trained PETS deploys directly as P2.

## Train any agent via the dispatcher

```bash
uv run python scripts/train.py --agent lewm    --steps 1000              # Stage A (JEPA pretraining)
uv run python scripts/train.py --agent lewm --stage b --steps 20000      # Stage B (heads on top of Stage A)
uv run python scripts/train.py --agent dreamer --steps 1000
uv run python scripts/train.py --agent pets    --steps 1000
```

## Smoke tests (no game required)

```bash
make test                                       # unittest regression suite; pytest not required
uv run python scripts/demo_state_vector.py        # primitives dict + new replay schema round-trip
uv run python scripts/demo_lewm_synthetic.py      # full JEPA stack on random tensors (MPS/CPU)
uv run python scripts/demo_pets_synthetic.py      # ensemble train + CEM planner end-to-end
uv run python scripts/demo_dreamer_synthetic.py   # full Dreamer (WM + actor + critic) train step (heavy on CPU/MPS)
```

## Running the head-to-head benchmark

The [reported results](#research-report-and-results) use the corrected
incoming-action Dreamer checkpoint saved at logged step 63,000,
`data/dreamer_checkpoint_incoming_v1.pt`, not the legacy 71k checkpoint.
They remain preliminary controller-level results, not a controlled
world-model comparison.

**Historical readiness audit.** The audit found
that historical Dreamer exports supplied outgoing actions to an RSSM that
expects incoming actions. The legacy 71k checkpoint has legacy/unknown
alignment and is strongly implicated by its exporter/cache provenance;
fixing the exporter cannot repair those weights. Corrected exports and
retraining were subsequently used for the reported Dreamer controller.
Do not substitute the legacy checkpoint when reproducing these results.
LeWM's pixel training is P1-only, so unchanged P2 deployment is not a
validated symmetric comparison.

Checkpoints used in the report (local artifacts, not included in Git):

| Model | Checkpoint | Native frame skip |
|---|---|---|
| LeWM, established v3 heads | `data/lewm_heads_checkpoint_stride5_m4_v3.pt` | 5 |
| Dreamer, corrected incoming-action alignment | `data/dreamer_checkpoint_incoming_v1.pt` | 1 |
| PETS, best validation snapshot | `data/pets_checkpoint_vsdreamer.pt` | 1 |

The two original replay mounts (`/media/jeovach/New Volume/leworldgaming`
and `/media/jeovach/Hoctap/leword-opponent`) were absent during the earlier
audit. At that time, the local `data/replay.h5` failed the
interleaved-perspective integrity check and held-out metrics could not be
reproduced. Do not substitute an invalid replay. Earlier bounded live
smoke runs also timed out on a heavily loaded host; those attempts were
**inconclusive**, not passed integration checks. The later completed
matches reported above do not establish held-out prediction quality.

Use **explicit trained checkpoint paths**; LeWM needs a Stage-B checkpoint.
First preview checkpoint identities and the four 60 Hz pairings per seed
without starting games: LeWM (fixed P1) vs Dreamer, LeWM (fixed P1) vs PETS,
and Dreamer/PETS in both player orders.

```bash
uv run python scripts/evaluate.py \
  --lewm-ckpt data/lewm_heads_checkpoint_stride5_m4_v3.pt \
  --dreamer-ckpt data/dreamer_checkpoint_incoming_v1.pt \
  --pets-ckpt data/pets_checkpoint_vsdreamer.pt \
  --out-dir data/benchmark_sync_60hz --seeds 0 1 2 --games 3 --device cuda \
  --pace sync --dry-run
```

Removing `--dry-run` enforces readiness checks before launching matches.
`--pace sync` is the **lockstep 60 Hz control-quality panel**: all sides
receive the current frame and choose a new action every raw frame while the
JVM waits. LeWM predicts five-frame latent transitions conditioned on raw
action blocks, then re-runs CEM every frame; Dreamer and PETS use their native
one-frame policies. This equalizes feedback decision opportunities, not
planner compute, training modality, or LeWM player side.

Repeat the same command with `--pace realtime` and a different output
directory for the **60 Hz deadline-feasibility panel**. Its JVM does not use
`--input-sync`, so late inference reaches the game late; do not combine its
win rates with the synchronous panel.

The runner uses Linux native rendering (`DISPLAY` defaults to `:1`).
Each match gets its own JVM; port 31415 must be free. The runner
never kills an existing server. With these settings this is **18 matches,
54 games**, not a smoke test. This is a new-run example, not the report's
early-stopped schedule, which requested five games per match and selected
nine completed matches for analysis. Checkpoint hashes, code revision/dirty status,
seeds, effective frame skips, raw round outcomes, and decision latency are
saved alongside logs. Existing output directories/results are never
overwritten, and a failed match aborts without writing a final summary.
Keep the source revision and checkpoints frozen for the entire run; the
runner rejects changes between matches. Execution cadence must match the
trained timestep: historical PETS/Dreamer frame-skip overrides do not.
Inference now restricts all three model controllers to static commandable
actions (including NEUTRAL), without changing network shapes or weights.
This corrects PETS/Dreamer's previous search over observation-only state
labels; historical scores from that behavior are not directly comparable.

For a single match on an already-running server, `scripts/self_play.py`
accepts `--seed` and `--output result.json`. For a bounded native integration
check, prefix `bash scripts/run_eval.sh ...` with
`JVM_EXTRA='-r 1 -f 120'` (one 120-frame round); **unset `JVM_EXTRA` for final
scores**.

This measures end-to-end controller performance, **not an equal-planner
world-model comparison**: Dreamer's actor, LeWM's planner and PETS's planner
have different compute and observation inputs. The synchronous panel's
reported budget overruns are informational; the real-time panel is the
deadline result. Seeds cover Python/NumPy/Torch, not JVM internals or
concurrent GPU scheduling.
The original shared-planner experiment and prediction/probe axes in
[`docs/benchmark_plan.md`](docs/benchmark_plan.md) are not implemented by this
runner.

## Layout

```
src/leworldgaming/   research code (env, agents, training, eval, data)
scripts/             runnable drivers
configs/             YAML configs
external/            vendored research code (le-wm, dreamerv3-torch)
docker/              JVM container (Linux training box)
vendor/              native FightingICE install (Mac, gitignored)
docs/                planning + research notes
paper/               research report PDF, LaTeX source, figures, and plotting scripts
```
