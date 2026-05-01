#!/usr/bin/env bash
set -euo pipefail

# MODE selects the JVM rendering / monitoring profile:
#   fast   — no rendering, ScreenData empty (state-vector agents). Default.
#   pixels — full rendering on Xvfb, pixels in ScreenData. No VNC.
#   watch  — full rendering + x11vnc on :5900 so you can watch the game on Mac.
#
# Also accepts the legacy LIGHTWEIGHT_MODE env var (1/0) for back-compat.
MODE="${MODE:-}"
if [ -z "$MODE" ]; then
    if [ "${LIGHTWEIGHT_MODE:-1}" = "0" ]; then
        MODE="pixels"
    else
        MODE="fast"
    fi
fi

case "$MODE" in
    fast)
        LIGHTWEIGHT_FLAG=("--lightweight-mode")
        START_VNC=0
        ;;
    pixels)
        LIGHTWEIGHT_FLAG=()
        START_VNC=0
        ;;
    watch)
        LIGHTWEIGHT_FLAG=()
        START_VNC=1
        ;;
    *)
        echo "[entrypoint] unknown MODE='$MODE' (expected fast|pixels|watch)" >&2
        exit 64
        ;;
esac

echo "[entrypoint] MODE=$MODE  lightweight=${#LIGHTWEIGHT_FLAG[@]}  vnc=$START_VNC"

# Always start Xvfb — even in fast mode, the JVM touches AWT during boot.
Xvfb :99 -screen 0 960x640x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &

for _ in $(seq 1 20); do
    if [ -e /tmp/.X11-unix/X99 ]; then break; fi
    sleep 0.1
done

export DISPLAY=:99

# Watch mode: expose Xvfb's framebuffer over VNC on :5900. Password protected
# only because macOS Screen Sharing refuses passwordless servers — the port
# is bound to 127.0.0.1 by docker so it's not actually network-exposed.
if [ "$START_VNC" = "1" ]; then
    VNC_PASSWORD="${VNC_PASSWORD:-watch}"
    PWFILE=/tmp/x11vnc.pw
    x11vnc -storepasswd "$VNC_PASSWORD" "$PWFILE" >/dev/null 2>&1
    x11vnc \
        -display :99 \
        -forever \
        -shared \
        -rfbauth "$PWFILE" \
        -rfbport 5900 \
        -quiet \
        -bg \
        -o /tmp/x11vnc.log \
        || echo "[entrypoint] x11vnc failed to start (continuing without it)"
    echo "[entrypoint] VNC: open vnc://localhost:5900 (password: $VNC_PASSWORD)"
fi

# `exec` so java is PID 1 — docker stop -> SIGTERM -> graceful JVM shutdown.
# /opt/lwjgl-natives/* are the linux-x86_64/aarch64 native JARs added in the
# Dockerfile. They're harmless in lightweight mode, required when rendering.
exec java \
    -cp "FightingICE.jar:./lib/*:/opt/lwjgl-natives/*" \
    Main \
    "${LIGHTWEIGHT_FLAG[@]}" \
    "$@"
