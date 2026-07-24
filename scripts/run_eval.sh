#!/bin/bash
# Launch a fresh JVM, wait for it to listen, run a self_play eval, then kill JVM.
# Usage: run_eval.sh <logfile> <extra self_play args...>
set -u
REPO=/home/jeovach/dev/leworldgaming/leworldgaming
LOG="$1"; shift
pkill -f "java.*Main" 2>/dev/null; sleep 5
cd "$REPO/vendor/fightingice"
DISPLAY=:1 java -cp 'FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/linux/amd64/*:./lib/grpc/*' \
  Main --limithp 400 400 --grey-bg --pyftg-mode --input-sync ${JVM_EXTRA:-} > /tmp/jvm_run.log 2>&1 &
JVM=$!
# wait for "listening" up to 60s
for i in $(seq 1 60); do
  grep -q "listening on 31415" /tmp/jvm_run.log && break
  kill -0 $JVM 2>/dev/null || { echo "JVM died early"; cat /tmp/jvm_run.log; exit 1; }
  sleep 1
done
sleep 3
echo "JVM up (pid $JVM), starting eval..."
cd "$REPO"
timeout 6000 uv run python scripts/self_play.py "$@" > "$LOG" 2>&1
echo "eval exit=$?"
kill $JVM 2>/dev/null
tail -25 "$LOG"
