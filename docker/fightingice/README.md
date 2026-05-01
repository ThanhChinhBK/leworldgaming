# DareFightingICE container (game engine only)

The FightingICE engine is a JVM app. Running it in Docker means we don't need to install a JDK on either Mac or the Linux box.

**Important:** the leworldgaming Python research code is **not** containerized — it runs natively under `uv` and connects to this container on `localhost:31415`.

## Three modes via `MODE`

| `MODE`   | Renders? | VNC?    | When to use                                         |
| -------- | -------- | ------- | --------------------------------------------------- |
| `fast`   | no       | no      | Default. State-vector training. Fastest.            |
| `pixels` | yes      | no      | Headless pixel collection (LeWM) on the Linux box.  |
| `watch`  | yes      | **yes** | Watch the game live on Mac via VNC.                 |

```bash
# default — fast, no pixels, no display
docker compose -f docker/fightingice/docker-compose.yml up

# pixels in ScreenData, no display (Linux box, headless)
MODE=pixels docker compose -f docker/fightingice/docker-compose.yml up

# pixels in ScreenData + VNC viewer on :5900 (monitor on Mac)
MODE=watch docker compose -f docker/fightingice/docker-compose.yml up
```

## Watching the game on macOS

```bash
MODE=watch docker compose -f docker/fightingice/docker-compose.yml up -d
open vnc://localhost:5900           # macOS Screen Sharing
# password: watch  (override with VNC_PASSWORD=... when starting the container)
```

macOS Screen Sharing requires a password even on local connections, so the entrypoint sets a default of `watch`. The VNC port is bound to `127.0.0.1` only, so the password is just a protocol formality — change it via `VNC_PASSWORD=mypw MODE=watch docker compose up` if you care. No XQuartz, no host-side X server. Window size: **960×640** (the FightingICE play area).

You'll see:
1. The "Waiting for AI" launcher screen until you start a game (e.g. `uv run python scripts/collect_data.py`).
2. The fight itself, animated at 60 FPS, when both AIs connect and the round begins.

Stop with `docker compose down` from the `docker/fightingice/` directory.

## What the image bakes in (vs upstream)

The upstream `ghcr.io/teamfightingice/fightingice:latest` is distroless and packaged for `--lightweight-mode` only. Render mode crashes on missing libs/assets. This image fixes that:

- **LWJGL native libraries** for both `linux-x86_64` and `linux-arm64`, downloaded from Maven Central at build time.
- **`resource-7.1.zip`** asset bundle (sprites, backgrounds) from the FightingICE GitHub release — upstream ships only `gSetting.txt` + `Motion.csv` per character.
- **Xvfb + GL + audio + font libs** so `DisplayManager`, `GraphicManager`, `SunFontManager`, and `SoundManager` all init cleanly.
- **`ALSOFT_DRIVERS=null`** env var so OpenAL doesn't fail on a missing audio device.
- **`x11vnc`** for the `watch` mode.

## Apple Silicon caveats

- The base `eclipse-temurin:17-jre-jammy` image is multi-arch, so the JVM runs natively on M-series Macs. No qemu emulation cost for state-vector mode.
- **Render mode + AI socket loop** has been observed to drop the AI client connection on macOS Docker (`Broken pipe` on the JVM side, `IncompleteReadError` on the Python side) — the GL pipeline is too slow under macOS Docker's networking layer to keep both AI controllers alive concurrently. State-vector mode (`MODE=fast`) is reliable on Mac. **`MODE=watch` is great for monitoring** but real pixel data collection (`MODE=pixels`) should run on the Linux+RTX 3080 box.
- `pyftg.Gateway.run_game()` does not return cleanly after `game_end` (waits forever on a trailing socket byte that never arrives). `scripts/collect_data.py` uses `--timeout` to cap each session and flush the buffer cleanly. After a hard kill, do `docker compose down && up` (not `restart`) to clear stale TCP state in the JVM's `SocketServer` thread.

## Game flags (passed via `command:` in compose)

- `--pyftg-mode` — required so the JVM expects external Python AI clients on port 31415.
- `--input-sync` — engine waits for both AI clients before stepping. Required for deterministic batched collection.
- `--limithp 400 400` — HP cap per player.
- `--lightweight-mode` is added by the entrypoint when `MODE=fast`.
