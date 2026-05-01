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

## Smoke tests (no game required)

```bash
uv run python scripts/demo_lewm_synthetic.py   # tiny JEPA training step on random tensors
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
