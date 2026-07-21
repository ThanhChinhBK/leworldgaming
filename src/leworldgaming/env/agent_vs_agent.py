"""Run two trained agents (or baselines) against each other directly.

Unlike ``FightingIceEnv`` (which bridges ONE python-controlled AI to an
external ``reset()``/``step()`` caller via queues), here BOTH sides are
self-driving: each ``_SelfDrivingAI`` calls its own agent's ``act(obs)``
synchronously inside ``processing()``. This is safe because pyftg already
runs ``processing()`` in a thread-pool executor per AI
(``await loop.run_in_executor(None, self.ai.processing)``), so two agents'
blocking inference calls (even GPU-bound) run concurrently without
stalling the asyncio event loop or each other.

Use this for agent-vs-agent tournaments (LeWM vs Dreamer, PETS vs LeWM,
etc.) via ``scripts/self_play.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.enums.action import Action
from pyftg.models.enums.status_code import StatusCode
from pyftg.models.frame_data import FrameData
from pyftg.models.game_data import GameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult
from pyftg.models.screen_data import ScreenData
from pyftg.protoc import service_pb2
from pyftg.socket.aio.ai_controller import AIController
from pyftg.socket.aio.gateway import Gateway
from pyftg.socket.aio.stream_controller import StreamController
from pyftg.socket.utils.asyncio import recv_data, send_data

from leworldgaming.env.fightingice_env import _to_pixel_tensor
from leworldgaming.env.spectator_recorder import SpectatorRecorder
from leworldgaming.env.state_vector import frame_to_obs_dict
from leworldgaming.utils.timing import FRAME_BUDGET_MS, FrameBudget

logger = logging.getLogger(__name__)


class ActingAgent(Protocol):
    def act(self, obs: dict[str, Any]) -> int: ...


@dataclass
class RoundOutcome:
    round_index: int
    hp_p1: float | None
    hp_p2: float | None
    winner: str | None  # "P1" | "P2" | "draw" | None (unknown)


@dataclass
class MatchResult:
    p1_name: str
    p2_name: str
    rounds: list[RoundOutcome] = field(default_factory=list)
    # Per-agent decision-latency stats (mean_ms/p95_ms/drop_rate/total),
    # keyed by p1_name/p2_name. Empty if latency wasn't recorded (shouldn't
    # happen via run_match(), but keeps this dataclass safely constructible
    # elsewhere without it).
    latency: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def wins_p1(self) -> int:
        return sum(1 for r in self.rounds if r.winner == "P1")

    @property
    def wins_p2(self) -> int:
        return sum(1 for r in self.rounds if r.winner == "P2")


class _SelfDrivingAI(AIInterface):
    """pyftg AI that drives itself: calls ``agent.act(obs)`` inline, no queues.

    Mirrors ``fightingice_env._BridgeAI`` but is fully self-contained since
    there's no external ``reset()``/``step()`` caller — both sides of the
    match are internal here.
    """

    def __init__(
        self,
        name: str,
        agent: ActingAgent,
        obs_mode: str,
        image_size: int,
        pixel_source: SpectatorRecorder | None,
        outcomes: list[RoundOutcome],
        max_hp: float = 400.0,
        max_energy: float = 300.0,
        frame_skip: int = 1,
    ) -> None:
        self._name = name
        self._agent = agent
        self._obs_mode = obs_mode
        self._image_size = image_size
        self._pixel_source = pixel_source
        self._outcomes = outcomes
        self._max_hp = max_hp
        self._max_energy = max_energy
        self._frame_skip = max(1, int(frame_skip))
        # Real-time decision-latency profiler (see
        # docs/lewm_stride2_vs_dreamer_2026-07-19.md — this was the missing
        # measurement needed to confirm/deny "heavier planner search loses
        # to GPU contention" instead of inferring it from win-rate deltas
        # alone). One decision = one ``agent.act()`` call, budgeted at
        # ``frame_skip`` raw frames' worth of wall-clock (mirrors
        # scripts/play.py's realtime-pace FrameBudget).
        self._latency = FrameBudget(budget_ms=FRAME_BUDGET_MS * self._frame_skip)

        self._cc = CommandCenter()
        self._key = Key()
        self._frame_data = FrameData()
        self._player_number = False
        self._skip_ctr = 0
        self._pending_action: Action | None = None
        self._need_reset = True
        self._round_index = 0
        # Diagnostics: track consecutive frames this AI emitted a "do
        # nothing" key (blank Key(), i.e. no button held) so we can log
        # *why* whenever it goes on long enough to look like the bot froze,
        # instead of silently sending blank input forever. See
        # ``_STUCK_WARN_FRAMES`` below.
        self._noop_streak = 0
        self._last_noop_reason: str | None = None

    def name(self) -> str:
        return self._name

    def is_blind(self) -> bool:
        return True

    def initialize(self, game_data: GameData, player_number: bool) -> None:
        self._player_number = player_number
        idx = 0 if player_number else 1
        if game_data.max_hps:
            self._max_hp = float(game_data.max_hps[idx])
        if game_data.max_energies:
            self._max_energy = float(game_data.max_energies[idx])
        logger.info("[%s] initialized as P%d", self._name, 1 if player_number else 2)

    def get_non_delay_frame_data(self, _frame_data: FrameData) -> None:
        pass

    def get_information(self, frame_data: FrameData, _is_control: bool) -> None:
        self._frame_data = frame_data

    def get_screen_data(self, _screen_data: ScreenData) -> None:
        pass

    def get_audio_data(self, _audio_data: AudioData) -> None:
        pass

    def _build_obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = frame_to_obs_dict(
            self._frame_data, self._player_number,
            max_hp=self._max_hp, max_energy=self._max_energy,
        )
        if self._obs_mode == "pixel":
            px = self._pixel_source.latest_pixels() if self._pixel_source else None
            obs["pixels"] = _to_pixel_tensor(px, self._image_size)
        return obs

    def _mark_noop(self, reason: str, frame_no: int) -> None:
        """Record that this frame produced a blank/no-op ``Key()`` and log a
        warning once the streak crosses ``_STUCK_WARN_FRAMES`` — this is the
        "bot has no action at all" symptom the user reported. Logging the
        *reason* (rather than just the symptom) lets us tell apart: waiting
        on frame data, a missing character (round transition), an agent
        exception, or the planner genuinely choosing to hold still.
        """
        if reason == self._last_noop_reason:
            self._noop_streak += 1
        else:
            self._noop_streak = 1
            self._last_noop_reason = reason
        if self._noop_streak == self._STUCK_WARN_FRAMES:
            logger.warning(
                "[%s] stuck: %d consecutive no-op frames (reason=%s) ending at frame %d",
                self._name, self._noop_streak, reason, frame_no,
            )
        elif self._noop_streak > 0 and self._noop_streak % (self._STUCK_WARN_FRAMES * 5) == 0:
            logger.warning(
                "[%s] still stuck: %d consecutive no-op frames (reason=%s) at frame %d",
                self._name, self._noop_streak, reason, frame_no,
            )

    def _clear_noop(self) -> None:
        if self._noop_streak >= self._STUCK_WARN_FRAMES:
            logger.info("[%s] resumed acting after %d no-op frames (reason=%s)",
                        self._name, self._noop_streak, self._last_noop_reason)
        self._noop_streak = 0
        self._last_noop_reason = None

    # ~0.5s at 60fps: long enough to rule out normal per-decision skip
    # windows (frame_skip is typically <=5) but short enough to catch a
    # genuine freeze quickly in a live match.
    _STUCK_WARN_FRAMES = 30

    def processing(self) -> None:
        fd = self._frame_data
        frame_no = fd.current_frame_number
        if fd.empty_flag or frame_no < 0:
            self._key = Key()
            self._mark_noop("empty_frame_data", frame_no)
            return
        own = fd.get_character(self._player_number)
        opp = fd.get_character(not self._player_number)
        if own is None or opp is None:
            self._key = Key()
            self._mark_noop("missing_character", frame_no)
            return

        self._cc.set_frame_data(fd, self._player_number)

        self._skip_ctr -= 1
        if self._skip_ctr <= 0:
            if self._need_reset:
                if hasattr(self._agent, "reset_episode"):
                    self._agent.reset_episode()
                self._need_reset = False
            obs = self._build_obs()
            try:
                with self._latency:
                    action_int = self._agent.act(obs)
            except Exception:
                # Never let an agent bug (bad tensor shape, NaNs, OOM, ...)
                # silently degrade to "no input forever" — log the full
                # traceback once per occurrence and fall back to NEUTRAL so
                # the match keeps going and the cause is still discoverable.
                logger.exception(
                    "[%s] agent.act() raised at frame %d; falling back to NEUTRAL",
                    self._name, frame_no,
                )
                action_int = 0  # IntAction.NEUTRAL == 0; Action.NEUTRAL.value is a str name
            self._pending_action = Action.from_int(int(action_int))
            logger.debug(
                "[%s] frame %d: action=%s (id=%d)",
                self._name, frame_no, self._pending_action.name, int(action_int),
            )
            self._skip_ctr = self._frame_skip

        if self._pending_action is None:
            self._key = Key()
            self._mark_noop("no_pending_action", frame_no)
            return
        skill_flag_before = self._cc.get_skill_flag()
        if not skill_flag_before:
            self._cc.command_call(self._pending_action.name)
        self._key = self._cc.get_skill_key()
        key_is_blank = not any(
            (self._key.A, self._key.B, self._key.C,
             self._key.U, self._key.R, self._key.D, self._key.L)
        )
        if key_is_blank:
            # command_call ran but produced no held key: either the chosen
            # action name doesn't map to a real command (should be rare now
            # that `_commandable_action_ids` restricts sampling — if this
            # fires a lot, that restriction has a gap) or the requested
            # action genuinely is NEUTRAL/a deliberate no-op.
            reason = (
                "neutral_action" if self._pending_action.name == "NEUTRAL"
                else f"uncommandable_action:{self._pending_action.name}"
            )
            self._mark_noop(reason, frame_no)
        else:
            self._clear_noop()

    def input(self) -> Key:
        return self._key

    def round_end(self, round_result: RoundResult) -> None:
        self._need_reset = True
        self._pending_action = None
        self._skip_ctr = 0
        # Only P1's AI records each round's outcome (avoids double-counting).
        if self._player_number:
            rem = getattr(round_result, "remaining_hps", None)
            hp_p1 = hp_p2 = None
            winner: str | None = None
            if rem and len(rem) >= 2:
                hp_p1, hp_p2 = float(rem[0]), float(rem[1])
                if hp_p1 > hp_p2:
                    winner = "P1"
                elif hp_p2 > hp_p1:
                    winner = "P2"
                else:
                    winner = "draw"
            self._round_index += 1
            self._outcomes.append(RoundOutcome(self._round_index, hp_p1, hp_p2, winner))

    def game_end(self) -> None:
        pass

    def close(self) -> None:
        pass

    def latency_stats(self) -> dict[str, float]:
        """Decision-latency summary for this side's ``agent.act()`` calls.

        ``mean_ms``/``p95_ms`` are wall-clock per decision; ``drop_rate`` is
        the fraction of decisions that exceeded ``frame_skip * 16.67ms``
        (i.e. would have arrived too late in a real-time realtime-pace
        match). ``total`` is the number of decisions timed.
        """
        hist = self._latency.history_ms
        if not hist:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "drop_rate": 0.0, "total": 0.0,
                     "budget_ms": self._latency.budget_ms}
        sorted_hist = sorted(hist)
        p95_idx = min(len(sorted_hist) - 1, int(round(0.95 * (len(sorted_hist) - 1))))
        return {
            "mean_ms": sum(hist) / len(hist),
            "p95_ms": sorted_hist[p95_idx],
            "drop_rate": self._latency.drop_rate,
            "total": float(self._latency.total),
            "budget_ms": self._latency.budget_ms,
        }


async def _run_match_async(
    p1_agent: ActingAgent,
    p2_agent: ActingAgent,
    *,
    p1_name: str = "AGENT_P1",
    p2_name: str = "AGENT_P2",
    p1_obs_mode: str = "state",
    p2_obs_mode: str = "state",
    p1_frame_skip: int = 1,
    p2_frame_skip: int = 1,
    host: str = "127.0.0.1",
    port: int = 31415,
    character: str = "ZEN",
    games: int = 1,
    image_size: int = 224,
) -> MatchResult:
    outcomes: list[RoundOutcome] = []
    needs_pixels = p1_obs_mode == "pixel" or p2_obs_mode == "pixel"
    spectator = SpectatorRecorder(image_size=image_size) if needs_pixels else None

    gateway = Gateway(host=host, port=port)
    ai_p1 = _SelfDrivingAI(
        p1_name, p1_agent, p1_obs_mode, image_size, spectator, outcomes,
        frame_skip=p1_frame_skip,
    )
    ai_p2 = _SelfDrivingAI(
        p2_name, p2_agent, p2_obs_mode, image_size, spectator, outcomes,
        frame_skip=p2_frame_skip,
    )
    gateway.register_ai(p1_name, ai_p1)
    gateway.register_ai(p2_name, ai_p2)

    reader, writer = await asyncio.open_connection(host, port)
    request = service_pb2.RunGameRequest(
        character_1=character, character_2=character,
        player_1=p1_name, player_2=p2_name,
        game_number=games,
    )
    await send_data(writer, b"\x02", with_header=False)
    await send_data(writer, request.SerializeToString())
    response_packet = await recv_data(reader)
    response = service_pb2.RunGameResponse()
    response.ParseFromString(response_packet)
    if response.status_code is StatusCode.FAILED:
        raise RuntimeError(f"JVM refused game: {response.response_message}")
    logger.info("game accepted: P1=%s P2=%s", p1_name, p2_name)

    ai_tasks = [
        asyncio.create_task(AIController(host, port, ai_p1, True).run()),
        asyncio.create_task(AIController(host, port, ai_p2, False).run()),
    ]
    spectator_task: asyncio.Task | None = None
    if spectator is not None:
        sctrl = StreamController(host, port, spectator, keep_alive=False)
        spectator_task = asyncio.create_task(sctrl.run())

    try:
        await asyncio.wait(ai_tasks, return_when=asyncio.ALL_COMPLETED)
        for task in ai_tasks:
            exc = task.exception()
            if exc is not None:
                logger.error("AI task failed: %r", exc)
    finally:
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

    return MatchResult(
        p1_name=p1_name, p2_name=p2_name, rounds=outcomes,
        latency={p1_name: ai_p1.latency_stats(), p2_name: ai_p2.latency_stats()},
    )


def run_match(
    p1_agent: ActingAgent,
    p2_agent: ActingAgent,
    **kwargs: Any,
) -> MatchResult:
    """Blocking entry point — runs a full agent-vs-agent match and returns results."""
    return asyncio.run(_run_match_async(p1_agent, p2_agent, **kwargs))
