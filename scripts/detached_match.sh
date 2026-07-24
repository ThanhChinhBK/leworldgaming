#!/bin/bash
# Fully-detached match runner: launches JVM (native window on :1) + OM-on match,
# survives parent shell exit via setsid. Logs to /tmp/match_*.log.
set -u
REPO=/home/jeovach/dev/leworldgaming/leworldgaming
pkill -9 -f "java.*Main" 2>/dev/null
sleep 3
cd "$REPO/vendor/fightingice"
DISPLAY=:1 java -cp 'FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/linux/amd64/*:./lib/grpc/*' \
  Main --limithp 400 400 --grey-bg --pyftg-mode --input-sync > /tmp/match_jvm.log 2>&1 &
JVM=$!
echo "jvm-pid=$JVM" > /tmp/match_status.txt
for i in $(seq 1 40); do
  sleep 1
  grep -q "listening on 31415" /tmp/match_jvm.log 2>/dev/null && break
done
echo "jvm-listening-at=${i}s" >> /tmp/match_status.txt
cd "$REPO"
echo "match-start=$(date +%H:%M:%S)" >> /tmp/match_status.txt
uv run python scripts/self_play.py \
  --p1 lewm --p1-ckpt data/lewm_heads_checkpoint_stride5_m4_v3.pt \
  --p2 dreamer --p2-ckpt data/dreamer_checkpoint.pt --p2-frame-skip 2 \
  --games 1 --device cuda --character ZEN \
  --opponent-model --opponent-model-strength 1.5 > /tmp/match_om.log 2>&1
echo "match-exit=$? match-end=$(date +%H:%M:%S)" >> /tmp/match_status.txt
kill $JVM 2>/dev/null
echo "DONE" >> /tmp/match_status.txt
