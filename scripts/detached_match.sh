#!/usr/bin/env bash
# Legacy convenience entry point; run_eval owns and cleans up only its own JVM.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_eval.sh /tmp/match_om.log \
  --p1 lewm --p1-ckpt data/lewm_heads_checkpoint_stride5_m4_v3.pt \
  --p2 dreamer --p2-ckpt data/dreamer_checkpoint.pt \
  --games 1 --device cuda --character ZEN \
  --opponent-model --opponent-model-strength 1.5 "$@"
