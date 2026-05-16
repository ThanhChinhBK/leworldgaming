"""Drive `RecordingAI` agents under pyftg's Gateway and write transitions
to an HDF5 replay buffer. Pass `--pixels` to also subscribe a spectator
stream that captures downsampled framebuffers.

Start the game first (`make game-native` on Mac, `make game` on Linux),
then run this in another terminal.

Supports playing against built-in JVM AIs (MctsAi23i, MctsAiZoning — the
ones shipped in DareFightingICE 7.1's data/ai/) by passing their class
name as ``--policy-p1`` or ``--policy-p2``. When a JVM AI name is
detected, no Python agent is created for that slot — in --pyftg-mode the
engine resolves the name against the classpath and instantiates the AI
server-side."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import time
from pathlib import Path

from pyftg.models.enums.status_code import StatusCode
from pyftg.protoc import service_pb2
from pyftg.socket.aio.ai_controller import AIController
from pyftg.socket.aio.gateway import Gateway
from pyftg.socket.aio.stream_controller import StreamController
from pyftg.socket.utils.asyncio import recv_data, send_data

from leworldgaming.data.replay_buffer import BufferConfig, ReplayBuffer
from leworldgaming.env.policies import make_policy
from leworldgaming.env.recording_ai import RecordingAI
from leworldgaming.env.spectator_recorder import SpectatorRecorder

# Built-in JVM AI class names shipped in vendor/fightingice/data/ai/*.jar.
# When one of these is passed as --policy-p1/p2, we don't create a Python
# agent for that slot — in --pyftg-mode the JVM resolves the name against
# the classpath and instantiates the AI in-process.
JVM_AIS = {"mctsai23i", "mctsaizoning"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=180, help="Number of games to play")
    parser.add_argument("--character", type=str, default="ZEN")
    parser.add_argument("--policy-p1", type=str, default="random",
                        help="P1 policy: 'random', 'noop', or a JVM AI class name "
                             "bundled in vendor/fightingice/data/ai/ "
                             "(MctsAi23i, MctsAiZoning)")
    parser.add_argument("--policy-p2", type=str, default="random",
                        help="P2 policy: 'random', 'noop', or a JVM AI class name "
                             "bundled in vendor/fightingice/data/ai/ "
                             "(MctsAi23i, MctsAiZoning)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31415)
    parser.add_argument("--out", type=str,
                        default="/media/jeovach/New Volume/leworldgaming/replay.h5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-record-p2", action="store_true",
                        help="Only record P1 (halves storage when P1=P2 self-play)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Hard wallclock timeout in seconds (default 300, scaled by --games)")
    parser.add_argument("--pixels", action="store_true",
                        help="Also record downsampled RGB frames (LeWM input). Requires "
                             "the JVM to be in render mode (native macOS, or docker MODE=pixels).")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Pixel side length in the buffer (default 224, ViT-friendly)")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("Data collection — %d games", args.games)
    logging.info("  output : %s", args.out)
    logging.info("  P1=%s  P2=%s  char=%s", args.policy_p1, args.policy_p2, args.character)
    logging.info("  pixels=%s  host=%s:%d", args.pixels, args.host, args.port)
    logging.info("=" * 60)

    t_start = time.monotonic()

    pixel_shape = (3, args.image_size, args.image_size) if args.pixels else None
    cfg = BufferConfig(path=args.out, pixel_shape=pixel_shape)
    buffer = ReplayBuffer(cfg)
    buffer.open()

    spectator: SpectatorRecorder | None = None
    spectator_task: asyncio.Task | None = None
    if args.pixels:
        spectator = SpectatorRecorder(image_size=args.image_size)

    p1_is_jvm = args.policy_p1.lower() in JVM_AIS
    p2_is_jvm = args.policy_p2.lower() in JVM_AIS

    if p1_is_jvm and p2_is_jvm:
        raise SystemExit(
            "Both P1 and P2 are JVM AIs — at least one must be a Python policy "
            "to record transitions."
        )

    gateway = Gateway(host=args.host, port=args.port)
    agent_names: list[str] = []

    if p1_is_jvm:
        agent_names.append(args.policy_p1)  # pass JVM name directly
    else:
        p1 = RecordingAI(
            name="LWG_P1",
            policy=make_policy(args.policy_p1, seed=args.seed),
            buffer=buffer,
            record=True,
            pixel_source=spectator,
            total_games=args.games,
        )
        gateway.register_ai("LWG_P1", p1)
        agent_names.append("LWG_P1")

    if p2_is_jvm:
        agent_names.append(args.policy_p2)  # pass JVM name directly
    else:
        p2 = RecordingAI(
            name="LWG_P2",
            policy=make_policy(args.policy_p2, seed=args.seed + 1),
            buffer=buffer,
            record=not args.no_record_p2,
            pixel_source=spectator,
            total_games=args.games,
        )
        gateway.register_ai("LWG_P2", p2)
        agent_names.append("LWG_P2")

    logging.info("agents: P1=%s  P2=%s", agent_names[0], agent_names[1])

    # --- Send RunGameRequest on a control connection ---
    reader, writer = await asyncio.open_connection(args.host, args.port)
    request = service_pb2.RunGameRequest(
        character_1=args.character, character_2=args.character,
        player_1=agent_names[0], player_2=agent_names[1],
        game_number=args.games,
    )
    await send_data(writer, b"\x02", with_header=False)
    await send_data(writer, request.SerializeToString())
    response_packet = await recv_data(reader)
    response = service_pb2.RunGameResponse()
    response.ParseFromString(response_packet)
    if response.status_code is StatusCode.FAILED:
        raise SystemExit(f"JVM refused game: {response.response_message}")
    logging.info("game accepted by server")

    # --- Start AI controller tasks (separate connections) ---
    ai_tasks: list[asyncio.Task] = []
    for i, name in enumerate(agent_names):
        agent = gateway.registered_agents.get(name)
        if agent is not None:
            ctrl_ai = AIController(args.host, args.port, agent, i == 0)
            ai_tasks.append(asyncio.create_task(ctrl_ai.run()))

    # FightingICE 7.x only ships pixels through the spectator path.
    if spectator is not None:
        ctrl = StreamController(args.host, args.port, spectator, keep_alive=False)
        spectator_task = asyncio.create_task(ctrl.run())

    # SIGINT/SIGTERM cancel AI tasks so the `finally` flush runs.
    loop = asyncio.get_running_loop()

    def _cancel() -> None:
        for t in ai_tasks:
            if not t.done():
                t.cancel()
        logging.info("signal received, cancelling AI tasks")

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _cancel)

    # Wait for AI controllers to finish (they exit on game_end from JVM).
    # The control connection may close early in render mode — ignore it.
    try:
        timeout = max(args.timeout, args.games * 120)
        done, pending = await asyncio.wait(
            ai_tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED,
        )
        for t in pending:
            logging.warning("AI task timed out — cancelling")
            t.cancel()
        # Re-raise any real errors (not IncompleteReadError from socket close).
        for t in done:
            exc = t.exception()
            if exc is not None and not isinstance(exc, asyncio.IncompleteReadError):
                raise exc
        if pending:
            logging.warning("game timeout (%ds) exceeded — closing", timeout)
    except asyncio.CancelledError:
        logging.info("game cancelled")
    finally:
        # Close control connection (may already be closed by JVM).
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()
        if spectator_task is not None and not spectator_task.done():
            spectator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await spectator_task
        if spectator is not None:
            spectator.close()
        await gateway.close()
        buffer.close()

    if spectator is not None:
        logging.info("spectator captured %d screen frames", spectator.frames_seen)
    elapsed = time.monotonic() - t_start
    mins, secs = divmod(int(elapsed), 60)
    logging.info(
        "Done — %d transitions, %d episodes in %dm%02ds -> %s",
        len(buffer),
        buffer.num_episodes,
        mins,
        secs,
        args.out,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
