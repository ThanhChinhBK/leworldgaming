#!/usr/bin/env bash
# Short "does-it-train" smoke run for the two benchmarked models — LeWM
# (JEPA, Stage A) and DreamerV3 — against an existing replay dataset.
#
# Unlike the synthetic demos (scripts/demo_*.py), this drives the real
# trainers over real collected HDF5 replays, so it exercises the full
# data path (DataReader directory loading, pixel + state-vector views,
# episode-aware sequence sampling) end to end. It is meant for validating
# the pipeline on a new dataset, not for producing usable checkpoints.
#
# Both trainers accept a directory of *.h5 files (see DataReader /
# dreamer_export) and sample from them jointly.
#
# Usage:
#   ./scripts/test_train.sh                       # defaults below
#   DATA_DIR=/path/to/h5s STEPS=20 ./scripts/test_train.sh
#   ./scripts/test_train.sh --steps 50            # forwarded to train.py
#
# Env knobs:
#   DATA_DIR    directory (or single .h5) of collected replays
#   STEPS       training steps per model
#   BATCH_SIZE  sequences per step (kept small so the smoke run is quick)
#   OUT_DIR     where test checkpoints/logs go (kept out of data/*.pt)
#   FORCE_CPU   set to 1 to hide the GPU (CUDA_VISIBLE_DEVICES="") and run on
#               CPU — useful when the installed torch wheel doesn't match the
#               local GPU arch (e.g. a Blackwell card needs a newer cu wheel).
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-/media/jeovach/New Volume/leworldgaming}"
STEPS="${STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUT_DIR="${OUT_DIR:-data/test_run}"

if [[ "${FORCE_CPU:-0}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    echo "FORCE_CPU=1 — hiding GPU, best_device() will fall back to CPU"
fi

mkdir -p "$OUT_DIR"

if [[ ! -e "$DATA_DIR" ]]; then
    echo "error: DATA_DIR not found: $DATA_DIR" >&2
    exit 1
fi

echo "=================================================================="
echo " test_train: smoke-training LeWM + Dreamer on real replays"
echo "   data    : $DATA_DIR"
echo "   steps   : $STEPS   batch: $BATCH_SIZE"
echo "   out     : $OUT_DIR"
echo "=================================================================="

echo
echo ">>> [1/2] LeWM (JEPA, Stage A) — consumes pixels"
uv run python scripts/train.py \
    --agent lewm \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --data-path "$DATA_DIR" \
    --ckpt-path "$OUT_DIR/lewm_test.pt" \
    "$@"

echo
echo ">>> [2/2] DreamerV3 (vector / proprio) — consumes state vectors"
uv run python scripts/train.py \
    --agent dreamer \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --data-path "$DATA_DIR" \
    --ckpt-path "$OUT_DIR/dreamer_test.pt" \
    "$@"

echo
echo "=================================================================="
echo " done. checkpoints written under: $OUT_DIR"
echo "=================================================================="
