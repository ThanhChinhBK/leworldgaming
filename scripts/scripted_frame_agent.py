"""Adapts the plain, cheap ``leworldgaming.env.policies.Policy`` callables
(random/aggressive/defensive/mixed — designed for the raw pyftg
``FrameData`` interface used by ``RecordingAI``) into the ``act(obs) ->
int`` interface that ``agent_vs_agent._SelfDrivingAI`` expects.

Those policies only need ``frame_data.get_character(player_number)`` for
mild heuristics (none of the current ones — random/aggressive/defensive —
actually inspect frame_data content; they're pure biased-random samplers),
so a dummy/None-shaped frame_data placeholder is enough here. If a future
policy needs actual frame content, extend this shim to reconstruct a
minimal frame_data-like object from ``obs`` instead.
"""

from __future__ import annotations

from typing import Any

from leworldgaming.env.policies import Policy


class ScriptedFrameAgent:
    """Wraps a ``Policy`` (frame_data, player_number) -> Action callable as
    an ``act(obs) -> int`` agent for ``run_match``/``_SelfDrivingAI``."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self.temporal_stride = 1

    def reset_episode(self) -> None:
        if hasattr(self._policy, "on_game_start"):
            self._policy.on_game_start()  # type: ignore[union-attr]

    def act(self, obs: dict[str, Any]) -> int:
        # None frame_data, player_number=True: fine because
        # random/aggressive/defensive/mixed never dereference frame_data.
        action = self._policy(None, True)
        return action.to_int()
