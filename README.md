# leworldgaming

Research scaffold comparing **LeWorldModel (JEPA)** against a **state-vector MBRL baseline** in **DareFightingICE**. Plan & motivation: [`docs/gemini_research.md`](docs/gemini_research.md).

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

**On Linux training box (docker):**

```bash
make game             # MODE=fast (no rendering)
make game-pixels      # MODE=pixels (LeWM data)
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

Output: `data/replay.h5` with datasets `state_vector / action / reward / done / hp_self / hp_opp / frame_idx / episode_starts` (and `pixels` when `--pixels` is set).

## Inspect data

```bash
uv run python scripts/extract_replay.py --stride 30   # dumps PNGs + metadata.csv to data/extracted/
open data/extracted/frames                            # browse in Finder
```

## Train LeWM (end-to-end)

The full pipeline: collect pixel data → train the JEPA world model → checkpoint.

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
| `noop` (alias `neutral`) | Always `Action.NEUTRAL` — passive baseline | Python (pyftg client) |
| `MctsAi23i` | Iteration-capped MCTS, the canonical DareFightingICE 7.x training opponent | JVM (loaded server-side from `vendor/fightingice/data/ai/MctsAi23i.jar`) |
| `MctsAiZoning` | MCTS variant with zoning heuristics | JVM (`MctsAiZoning.jar`) |

When a JVM AI name is passed, `collect_data.py` does **not** spin up a Python agent for that slot — in `--pyftg-mode` the engine resolves the class name against `data/ai/*.jar` and instantiates it in-process. At least one slot must be a Python policy so transitions get recorded.

> Older AIs (`MctsAi`, `KickAI`, `Sandbox`, `Thunder`, `BlindAI`, `ErheaPi`) are **not** bundled in DareFightingICE 7.1 — only the two MCTS jars above are. To use others, drop their compiled `.jar` into `vendor/fightingice/data/ai/` and add the class name to `JVM_AIS` in `scripts/collect_data.py`.

**Examples:**

```bash
# Random P1 vs the canonical MCTS opponent (BlindAI-paper recipe)
uv run python scripts/collect_data.py --games 30 --pixels --policy-p2 MctsAi23i

# Random P1 vs MCTS-with-zoning (different distribution of states)
uv run python scripts/collect_data.py --games 30 --pixels --policy-p2 MctsAiZoning

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
| `batch_size` | 16 | Sequences per step (each = `history_size+1` frames through ViT) |
| `lr` | 3e-4 | AdamW learning rate |
| `sigreg_lambda` | 0.1 | Weight of SIGReg anti-collapse regularizer |

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
- **|z|** — mean embedding L2 norm; healthy range ≈ √latent_dim ≈ **15–16** for dim=256. Collapse → 0, explosion → >>16.
- **grad** — gradient norm after clipping (should stay ≤ `grad_clip`)

### 4. Checkpoint & inference

Training saves to `data/lewm_checkpoint.pt` (configurable via `ckpt_path`). Load for inference:

```python
from leworldgaming.agents.lewm.agent import LewmAgent

agent = LewmAgent(device="mps")  # or "cuda"
agent.load("data/lewm_checkpoint.pt")  # rebuilds architecture from stored config
action = agent.act({"pixels": frame_tensor})  # (3, 224, 224) float
```

### Architecture (22.98M params at default config)

```
ViT-12 encoder (5.57M) → Projector (1.05M) → AR Predictor (14.97M) → pred_proj (1.05M)
                                                ↑ conditioned on ActionEncoder (0.32M)
```

The AR predictor uses causal self-attention + AdaLN-zero conditioning on per-step action embeddings, so one forward pass yields `T` parallel next-step predictions during training.

## Smoke tests (no game required)

```bash
uv run python scripts/demo_lewm_synthetic.py   # full JEPA stack on random tensors (MPS/CPU)
uv run python scripts/demo_state_vector.py     # state-vector + replay-buffer round trip
```

## Layout

```
src/leworldgaming/   research code (env, agents, training, eval, data)
scripts/             runnable drivers
configs/             YAML configs
external/            vendored research code (le-wm, dreamerv3-torch)
docker/              JVM container (Linux training box)
vendor/              native FightingICE install (Mac, gitignored)
docs/                planning + research notes
```
