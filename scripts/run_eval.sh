#!/usr/bin/env bash
# Run one match on a fresh JVM; never terminate an unrelated process.
# EVAL_PACE=sync waits for every action; EVAL_PACE=realtime uses the JVM clock.
# Usage: run_eval.sh <logfile> <self_play args...>
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${1:?usage: run_eval.sh <logfile> <self_play args...>}"; shift
mkdir -p "$(dirname "$LOG")"
LOG="$(cd "$(dirname "$LOG")" && pwd)/$(basename "$LOG")"
JVM_LOG="${LOG}.jvm.log"
if [[ -e "$LOG" || -e "$JVM_LOG" ]]; then
  echo "Refusing to overwrite existing logs: $LOG or $JVM_LOG" >&2
  exit 1
fi
JVM=""
cleanup() {
  if [[ -n "$JVM" ]]; then
    kill "$JVM" 2>/dev/null || true
    wait "$JVM" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$REPO"
uv run python - <<'PY'
import socket
with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", 31415)) == 0:
        raise SystemExit("Port 31415 is occupied; refusing to stop another JVM.")
PY
cd "$REPO/vendor/fightingice"
# JVM_EXTRA is a whitespace-separated list of JVM game flags, not shell code.
case "${EVAL_PACE:-sync}" in
  sync) pace_args=(--input-sync) ;;
  realtime) pace_args=() ;;
  *) echo "EVAL_PACE must be sync or realtime" >&2; exit 2 ;;
esac
read -r -a extra <<< "${JVM_EXTRA:-}"
DISPLAY="${DISPLAY:-:1}" java \
  -cp 'FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/linux/amd64/*:./lib/grpc/*' \
  Main --limithp 400 400 --grey-bg --pyftg-mode "${pace_args[@]}" "${extra[@]}" \
  > "$JVM_LOG" 2>&1 &
JVM=$!
ready=false
for ((i=0; i<60; i++)); do
  if ! kill -0 "$JVM" 2>/dev/null; then
    echo "JVM died early; see $JVM_LOG" >&2
    exit 1
  fi
  if [[ -f "$JVM_LOG" ]] && grep -q "listening on 31415" "$JVM_LOG"; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "JVM startup timed out; see $JVM_LOG" >&2
  exit 1
fi
cd "$REPO"
status=0
timeout "${EVAL_TIMEOUT:-6000}" uv run python scripts/self_play.py "$@" > "$LOG" 2>&1 || status=$?
tail -25 "$LOG"
exit "$status"
