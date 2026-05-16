"""Recording AI — pyftg `AIInterface` impl that picks actions via a Policy
and writes every (state_vector, action, reward, hp) to a `ReplayBuffer`."""

from __future__ import annotations

import logging
import time
from typing import Protocol

import numpy as np
from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.enums.action import Action
from pyftg.models.frame_data import FrameData
from pyftg.models.game_data import GameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult
from pyftg.models.screen_data import ScreenData

from leworldgaming.data.replay_buffer import ReplayBuffer
from leworldgaming.env.policies import Policy
from leworldgaming.env.state_vector import frame_to_obs_dict

logger = logging.getLogger(__name__)


class PixelSource(Protocol):
    """Latest framebuffer as (3,H,W) uint8. Implemented by `SpectatorRecorder`."""

    def latest_pixels(self) -> np.ndarray | None: ...


class RecordingAI(AIInterface):
    def __init__(
        self,
        name: str,
        policy: Policy,
        buffer: ReplayBuffer | None = None,
        record: bool = True,
        pixel_source: PixelSource | None = None,
        total_games: int = 1,
    ) -> None:
        self._name = name
        self._policy = policy
        self._buffer = buffer
        self._record = record and (buffer is not None)
        self._pixel_source = pixel_source
        self._total_games = total_games
        self._cc = CommandCenter()
        self._key = Key()
        self._frame_data = FrameData()
        self._player_number = False
        self._max_hp = 400.0
        self._max_energy = 300.0
        self._prev_hp_self: int | None = None
        self._prev_hp_opp: int | None = None
        self._steps_in_episode = 0
        self._game_index = 0
        self._round_index = 0
        self._t_start: float | None = None
        self._t_game_start: float = 0.0

    def name(self) -> str:
        return self._name

    def is_blind(self) -> bool:
        # FightingICE 7.x's per-AI socket never ships ScreenData regardless
        # of this flag (verified by disassembling service.SocketPlayer).
        # Pixels come from a parallel spectator stream via `pixel_source`.
        return True

    def initialize(self, game_data: GameData, player_number: bool) -> None:
        self._player_number = player_number
        idx = 0 if player_number else 1
        self._max_hp = float(game_data.max_hps[idx]) if game_data.max_hps else 400.0
        self._max_energy = (
            float(game_data.max_energies[idx]) if game_data.max_energies else 300.0
        )
        now = time.monotonic()
        if self._t_start is None:
            self._t_start = now
        self._t_game_start = now
        if hasattr(self._policy, "on_game_start"):
            self._policy.on_game_start()  # type: ignore[union-attr]
        logger.info(
            "[%s] initialized as P%d (max_hp=%.0f max_energy=%.0f)",
            self._name,
            1 if player_number else 2,
            self._max_hp,
            self._max_energy,
        )

    def get_non_delay_frame_data(self, frame_data: FrameData) -> None:
        pass

    def get_information(self, frame_data: FrameData, is_control: bool) -> None:
        self._frame_data = frame_data

    def get_screen_data(self, screen_data: ScreenData) -> None:
        pass

    def get_audio_data(self, audio_data: AudioData) -> None:
        pass

    def processing(self) -> None:
        if self._frame_data.empty_flag or self._frame_data.current_frame_number < 0:
            self._key = Key()
            return

        own = self._frame_data.get_character(self._player_number)
        opp = self._frame_data.get_character(not self._player_number)
        if own is None or opp is None:
            self._key = Key()
            return

        action: Action = self._policy(self._frame_data, self._player_number)

        if self._record and self._buffer is not None:
            obs = frame_to_obs_dict(
                self._frame_data,
                self._player_number,
                max_hp=self._max_hp,
                max_energy=self._max_energy,
            )
            if self._prev_hp_self is None:
                reward = 0.0
            else:
                # Damage dealt minus damage taken, normalized by max HP.
                damage_dealt = self._prev_hp_opp - opp.hp
                damage_taken = self._prev_hp_self - own.hp
                reward = float(damage_dealt - damage_taken) / max(self._max_hp, 1.0)
            pixels = self._pixel_source.latest_pixels() if self._pixel_source else None
            self._buffer.add(
                obs_dict=obs,
                action=action.to_int(),
                reward=reward,
                done=False,
                is_first=(self._steps_in_episode == 0),
                pixels=pixels,
            )
            self._prev_hp_self = own.hp
            self._prev_hp_opp = opp.hp
            self._steps_in_episode += 1

        self._cc.set_frame_data(self._frame_data, self._player_number)
        if not self._cc.get_skill_flag():
            self._cc.command_call(action.name)
        self._key = self._cc.get_skill_key()

    def input(self) -> Key:
        return self._key

    def round_end(self, round_result: RoundResult) -> None:
        if self._record and self._buffer is not None and self._steps_in_episode > 0:
            self._buffer.end_episode()
        self._round_index += 1
        steps = self._steps_in_episode
        self._prev_hp_self = None
        self._prev_hp_opp = None
        self._steps_in_episode = 0
        buf_len = len(self._buffer) if self._buffer else 0
        buf_eps = self._buffer.num_episodes if self._buffer else 0
        logger.info(
            "[%s] round %d end  |  %d steps this round  |  buffer: %d transitions, %d episodes",
            self._name, self._round_index, steps, buf_len, buf_eps,
        )

    def game_end(self) -> None:
        self._game_index += 1
        self._round_index = 0
        now = time.monotonic()
        game_secs = now - self._t_game_start
        self._t_game_start = now

        buf_len = len(self._buffer) if self._buffer else 0
        buf_eps = self._buffer.num_episodes if self._buffer else 0

        elapsed = now - (self._t_start or now)
        games_per_min = self._game_index / max(elapsed, 1) * 60
        remaining = self._total_games - self._game_index
        eta_secs = int(remaining / max(games_per_min, 0.01) * 60)
        eta_m, eta_s = divmod(eta_secs, 60)

        logger.info(
            "[%s] game %d/%d complete (%.1fs)  |  %.1f games/min  ETA %dm%02ds"
            "  |  buffer: %d transitions, %d episodes",
            self._name, self._game_index, self._total_games, game_secs,
            games_per_min, eta_m, eta_s, buf_len, buf_eps,
        )

    def close(self) -> None:
        pass
