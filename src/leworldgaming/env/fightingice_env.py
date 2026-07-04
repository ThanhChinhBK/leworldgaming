"""Synchronous ``reset()/step()`` wrapper around the live DareFightingICE game.

pyftg is **callback-driven**: its ``AIController`` calls our AI's
``processing()`` once per controllable frame and reads back ``input()``.
``AIController`` runs ``processing()`` in a thread-pool executor
(``await loop.run_in_executor(None, self.ai.processing)``), so blocking
inside ``processing()`` does **not** stall the event loop — the pixel
spectator stream keeps decoding concurrently. That lets us invert the push
API into a synchronous pull API with a two-queue handshake:

    env.step(action)  --action-->  _act_q  -->  _BridgeAI.processing()
    env.step(...)     <--obs-----  _obs_q   <--  _BridgeAI.processing()

The pyftg gateway runs on its own asyncio loop in a background thread
(``_thread_main``); ``_BridgeAI`` lives there. ``FightingIceEnv`` runs on
the caller's thread. ``queue.Queue`` is the thread-safe bridge.

Start the JVM game first (``make game-native`` on Mac, ``make game`` /
``make game-pixels`` on Linux), then drive this from Python — see
``scripts/play.py``. Model-free: exercise it today with ``--agent random``.

Episode == one round. ``step`` returns ``terminated=True`` at round end;
``reset`` then blocks until the next round's first frame. When all
``cfg.games`` games finish, ``reset`` returns ``(None, {"match_over": True})``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
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

from leworldgaming.env.policies import make_policy
from leworldgaming.env.recording_ai import RecordingAI
from leworldgaming.env.spectator_recorder import SpectatorRecorder
from leworldgaming.env.state_vector import frame_to_obs_dict

logger = logging.getLogger(__name__)

# Python-side policy names handled by env.policies.make_policy. Anything else
# passed as the opponent is treated as a JVM AI class name (resolved
# server-side from vendor/fightingice/data/ai/*.jar, e.g. MctsAi23i).
_PYTHON_POLICIES = {"random", "noop", "neutral", "aggressive", "defensive", "mixed"}


@dataclass
class EnvConfig:
    host: str = "127.0.0.1"
    port: int = 31415
    character: str = "ZEN"
    obs_mode: str = "state"  # "state" (PETS/Dreamer) | "pixel" (LeWM; adds obs["pixels"])
    agent_player: str = "P1"  # which slot the agent controls: "P1" | "P2"
    opponent: str = "MctsAi23i"  # JVM AI class name, or a python policy name
    games: int = 1  # number of games to request from the JVM (each = several rounds)
    frame_skip: int = 1  # decide every N frames; the chosen action repeats for N (action-repeat)
    image_size: int = 224
    seed: int = 0
    max_hp: float = 400.0
    max_energy: float = 300.0


def _to_pixel_tensor(px: np.ndarray | None, image_size: int):
    """uint8 (3,H,W) framebuffer -> float32 tensor in [-1, 1] (training norm).

    Matches ``training/_replay_utils.to_device_seq``: ``x/127.5 - 1``. Left on
    CPU; ``LewmAgent.act`` moves it to the model device.
    """
    import torch

    arr = np.zeros((3, image_size, image_size), dtype=np.uint8) if px is None else px
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=torch.float32).div_(127.5).sub_(1.0)


class _BridgeAI(AIInterface):
    """pyftg AI that hands each frame's observation to ``FightingIceEnv`` and
    blocks (in the executor thread) until the env supplies an action."""

    def __init__(
        self,
        name: str,
        obs_q: queue.Queue,
        act_q: queue.Queue,
        obs_mode: str,
        image_size: int,
        pixel_source: SpectatorRecorder | None = None,
        max_hp: float = 400.0,
        max_energy: float = 300.0,
        frame_skip: int = 1,
    ) -> None:
        self._name = name
        self._obs_q = obs_q
        self._act_q = act_q
        self._obs_mode = obs_mode
        self._image_size = image_size
        self._pixel_source = pixel_source
        self._max_hp = max_hp
        self._max_energy = max_energy
        self._frame_skip = max(1, int(frame_skip))

        self._cc = CommandCenter()
        self._key = Key()
        self._frame_data = FrameData()
        self._player_number = False
        self._prev_hp_self: int | None = None
        self._prev_hp_opp: int | None = None
        self._decisions = 0
        self._skip_ctr = 0  # counts down to the next decision frame
        self._pending_action: Action | None = None
        self._last_obs: dict[str, Any] | None = None

    def name(self) -> str:
        return self._name

    def is_blind(self) -> bool:
        # FightingICE 7.x never ships ScreenData on the AI socket; pixels come
        # from the parallel spectator stream (same as RecordingAI).
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

    def processing(self) -> None:
        fd = self._frame_data
        if fd.empty_flag or fd.current_frame_number < 0:
            self._key = Key()
            return
        own = fd.get_character(self._player_number)
        opp = fd.get_character(not self._player_number)
        if own is None or opp is None:
            self._key = Key()
            return

        self._cc.set_frame_data(fd, self._player_number)

        # Decision frame: surface the obs (HP-delta accumulated since the last
        # decision) and block for the agent's action. Skipped frames fall
        # through and keep re-issuing the pending action (action-repeat).
        self._skip_ctr -= 1
        if self._skip_ctr <= 0:
            obs = self._build_obs()
            if self._prev_hp_self is None:
                reward = 0.0
            else:
                damage_dealt = self._prev_hp_opp - opp.hp
                damage_taken = self._prev_hp_self - own.hp
                reward = float(damage_dealt - damage_taken) / max(self._max_hp, 1.0)
            self._prev_hp_self = own.hp
            self._prev_hp_opp = opp.hp
            info = {
                "is_first": self._decisions == 0,
                "frame": int(fd.current_frame_number),
                "hp_self": int(own.hp),
                "hp_opp": int(opp.hp),
            }
            self._decisions += 1
            self._last_obs = obs

            self._obs_q.put(("step", obs, reward, False, info))
            action_int = self._act_q.get()
            if action_int is None:  # close() sentinel — stop driving inputs.
                self._key = Key()
                return
            self._pending_action = Action.from_int(int(action_int))
            self._skip_ctr = self._frame_skip

        if self._pending_action is None:
            self._key = Key()
            return
        if not self._cc.get_skill_flag():
            self._cc.command_call(self._pending_action.name)
        self._key = self._cc.get_skill_key()

    def input(self) -> Key:
        return self._key

    def round_end(self, round_result: RoundResult) -> None:
        # No processing() fires for the terminal frame; synthesize the
        # terminal transition so the pending env.step() returns done=True.
        reward = 0.0
        rem = getattr(round_result, "remaining_hps", None)
        if rem and self._prev_hp_self is not None and len(rem) >= 2:
            idx = 0 if self._player_number else 1
            hp_self, hp_opp = float(rem[idx]), float(rem[1 - idx])
            damage_dealt = self._prev_hp_opp - hp_opp
            damage_taken = self._prev_hp_self - hp_self
            reward = float(damage_dealt - damage_taken) / max(self._max_hp, 1.0)
        info: dict[str, Any] = {"terminal": True}
        if rem and len(rem) >= 2:
            idx = 0 if self._player_number else 1
            info["hp_self"] = float(rem[idx])
            info["hp_opp"] = float(rem[1 - idx])
            info["win"] = float(rem[idx]) > float(rem[1 - idx])
        self._obs_q.put(("round_end", self._last_obs, reward, True, info))
        self._prev_hp_self = None
        self._prev_hp_opp = None
        self._decisions = 0
        self._skip_ctr = 0
        self._pending_action = None

    def game_end(self) -> None:
        self._obs_q.put(("game_end", None, 0.0, True, {"game_end": True}))

    def close(self) -> None:
        self._obs_q.put(("close", None, 0.0, True, {"match_over": True}))


class FightingIceEnv:
    """Synchronous Gym-style interface over a live DareFightingICE match.

    ``obs`` is the primitives dict from ``frame_to_obs_dict`` (consumed
    directly by ``PETSAgent`` / vector-mode Dreamer). In ``obs_mode="pixel"``
    it also carries ``obs["pixels"]`` — a float32 ``(3,H,W)`` tensor in
    ``[-1, 1]`` for ``LewmAgent``.
    """

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        self._obs_q: queue.Queue = queue.Queue()
        self._act_q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = False
        self._done = True
        self.match_over = False
        self._bridge: _BridgeAI | None = None
        self._spectator: SpectatorRecorder | None = None

    # -- lifecycle ---------------------------------------------------------

    def _start_match(self) -> None:
        self._thread = threading.Thread(
            target=self._thread_main, name="fightingice-match", daemon=True,
        )
        self._thread.start()
        self._started = True

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._match_main())
        except Exception as exc:  # surface to the waiting reset()/step()
            logger.exception("match thread crashed")
            self._obs_q.put(("error", None, 0.0, True, {"error": repr(exc)}))

    async def _match_main(self) -> None:
        cfg = self.cfg
        agent_is_p1 = cfg.agent_player.upper() == "P1"
        opp_is_jvm = cfg.opponent.lower() not in _PYTHON_POLICIES

        if cfg.obs_mode == "pixel":
            self._spectator = SpectatorRecorder(image_size=cfg.image_size)

        self._bridge = _BridgeAI(
            "LWG_AGENT", self._obs_q, self._act_q,
            obs_mode=cfg.obs_mode, image_size=cfg.image_size,
            pixel_source=self._spectator,
            max_hp=cfg.max_hp, max_energy=cfg.max_energy,
            frame_skip=cfg.frame_skip,
        )

        gateway = Gateway(host=cfg.host, port=cfg.port)
        gateway.register_ai("LWG_AGENT", self._bridge)

        opp_ai = None
        if not opp_is_jvm:
            opp_ai = RecordingAI(
                name="LWG_OPP",
                policy=make_policy(cfg.opponent, seed=cfg.seed + 1),
                buffer=None, record=False,
            )
            gateway.register_ai("LWG_OPP", opp_ai)
        opp_name = cfg.opponent if opp_is_jvm else "LWG_OPP"

        # agent_names[0] -> P1, [1] -> P2 (AIController player_number = index==0).
        agent_names = ["LWG_AGENT", opp_name] if agent_is_p1 else [opp_name, "LWG_AGENT"]

        # --- request the game on a control connection ---
        reader, writer = await asyncio.open_connection(cfg.host, cfg.port)
        request = service_pb2.RunGameRequest(
            character_1=cfg.character, character_2=cfg.character,
            player_1=agent_names[0], player_2=agent_names[1],
            game_number=cfg.games,
        )
        await send_data(writer, b"\x02", with_header=False)
        await send_data(writer, request.SerializeToString())
        response_packet = await recv_data(reader)
        response = service_pb2.RunGameResponse()
        response.ParseFromString(response_packet)
        if response.status_code is StatusCode.FAILED:
            raise RuntimeError(f"JVM refused game: {response.response_message}")
        logger.info("game accepted: P1=%s P2=%s", agent_names[0], agent_names[1])

        # --- start AI controllers (one per python AI) ---
        ai_tasks: list[asyncio.Task] = []
        for i, name in enumerate(agent_names):
            agent = gateway.registered_agents.get(name)
            if agent is not None:  # None => JVM AI, driven server-side
                ctrl = AIController(cfg.host, cfg.port, agent, i == 0)
                ai_tasks.append(asyncio.create_task(ctrl.run()))

        spectator_task: asyncio.Task | None = None
        if self._spectator is not None:
            sctrl = StreamController(cfg.host, cfg.port, self._spectator, keep_alive=False)
            spectator_task = asyncio.create_task(sctrl.run())

        try:
            await asyncio.wait(ai_tasks, return_when=asyncio.ALL_COMPLETED)
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()
            if spectator_task is not None and not spectator_task.done():
                spectator_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await spectator_task
            if self._spectator is not None:
                self._spectator.close()
            await gateway.close()

    # -- Gym-style API -----------------------------------------------------

    def reset(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Block until the next round's first frame.

        Returns ``(obs, info)``. When the whole match is over returns
        ``(None, {"match_over": True})`` — the caller should stop looping.
        """
        if not self._started:
            self._start_match()
        if self.match_over:
            return None, {"match_over": True}

        # Skip any pending terminal markers; wait for the next live frame.
        while True:
            kind, obs, _, _, info = self._obs_q.get()
            if kind == "step":
                self._done = False
                return obs, info
            if kind in ("close", "error"):
                self.match_over = True
                return None, info
            # 'game_end' between games is followed by the next round's 'step';
            # 'round_end' was already consumed by the prior step — keep waiting.

    def step(self, action: int) -> tuple[dict[str, Any] | None, float, bool, bool, dict[str, Any]]:
        """Apply ``action`` (int in ``[0, NUM_ACTIONS)``) for one frame."""
        if self._done:
            raise RuntimeError("step() after episode end — call reset() first")
        self._act_q.put(int(action))
        kind, obs, reward, _, info = self._obs_q.get()
        if kind == "step":
            return obs, reward, False, False, info
        if kind == "round_end":
            self._done = True
            return obs, reward, True, False, info  # obs = last live frame
        # game_end / close / error arriving here ends the episode + match.
        self._done = True
        self.match_over = True
        return None, reward, True, False, info

    def close(self) -> None:
        # Unblock a parked processing() so the match thread can wind down.
        with contextlib.suppress(Exception):
            self._act_q.put(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
