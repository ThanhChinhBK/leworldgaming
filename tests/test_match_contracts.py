from __future__ import annotations

import asyncio
import queue
import threading
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from pyftg.models.enums.status_code import StatusCode
from pyftg.protoc import service_pb2

from leworldgaming.env import agent_vs_agent as ava
from leworldgaming.env import fightingice_env as fie
from leworldgaming.env.match_contracts import PixelReader, run_controllers


def writer_mock():
    return SimpleNamespace(close=Mock(), wait_closed=AsyncMock())


def network_mocks(stack, module, *, rejected=False):
    writer = writer_mock()
    stack.enter_context(patch.object(
        module.asyncio, "open_connection", AsyncMock(return_value=(Mock(), writer)),
    ))
    stack.enter_context(patch.object(module, "send_data", AsyncMock()))
    response = service_pb2.RunGameResponse(
        status_code=StatusCode.FAILED.value if rejected else 0,
        response_message="rejected" if rejected else "accepted",
    )
    stack.enter_context(patch.object(
        module, "recv_data", AsyncMock(return_value=response.SerializeToString()),
    ))
    gateway_close = stack.enter_context(patch.object(module.Gateway, "close", AsyncMock()))
    return writer, gateway_close


class ConfigTests(unittest.TestCase):
    def test_env_rejects_invalid_config(self):
        invalid = [
            {"games": 0}, {"games": -1}, {"games": 1.5}, {"games": True},
            {"frame_skip": 0}, {"image_size": -1}, {"obs_mode": "pixels"},
            {"agent_player": "P3"}, {"opponent": ""}, {"opponent": "LWG_AGENT"},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fie.EnvConfig(**kwargs)
        self.assertEqual(fie.EnvConfig(agent_player="p2").agent_player, "p2")

    def test_match_rejects_invalid_config_before_connecting(self):
        invalid = [
            {"games": 0}, {"games": -1}, {"games": True}, {"games": 1.5},
            {"p1_frame_skip": 0}, {"p2_frame_skip": -1}, {"image_size": 0},
            {"p1_obs_mode": "pixels"}, {"p2_obs_mode": "other"},
            {"p1_name": "same", "p2_name": "same"}, {"p1_name": ""},
        ]
        with patch.object(ava.asyncio, "open_connection", AsyncMock()) as connect:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    ava.run_match(Mock(), Mock(), **kwargs)
            connect.assert_not_called()


class AgentTests(unittest.TestCase):
    def make_ai(self, agent=None, **kwargs):
        ai = ava._SelfDrivingAI("test", agent or Mock(), "state", 16, None, [], **kwargs)
        ai._frame_data = Mock(empty_flag=False, current_frame_number=1)
        ai._cc = Mock()
        ai._cc.get_skill_key.return_value = ava.Key()
        ai._build_obs = Mock(return_value={})
        return ai

    def test_agent_act_failure_is_not_neutral(self):
        failure = ValueError("bad tensor")
        ai = self.make_ai(Mock(act=Mock(side_effect=failure)))
        with self.assertLogs(ava.logger, level="ERROR"), self.assertRaises(ValueError) as caught:
            ai.processing()
        self.assertIs(caught.exception, failure)
        self.assertIsNone(ai._pending_action)

    def test_agent_reset_failure_propagates(self):
        ai = self.make_ai(Mock(reset_episode=Mock(side_effect=ValueError("bad reset"))))
        with self.assertRaisesRegex(ValueError, "bad reset"):
            ai.processing()
        ai._agent.act.assert_not_called()

    def test_recording_and_frame_skip_preserved(self):
        buffer = Mock()
        ai = self.make_ai(Mock(act=Mock(return_value=0)), record_buffer=buffer, frame_skip=2)
        own = SimpleNamespace(hp=100)
        opp = SimpleNamespace(hp=90)
        ai._frame_data.get_character.side_effect = [own, opp, own, opp]
        with patch.object(ava, "frame_to_obs_dict", return_value={}):
            ai.processing()
            ai.processing()
        ai._agent.act.assert_called_once_with({})
        self.assertEqual(buffer.add.call_count, 2)
        self.assertTrue(buffer.add.call_args_list[0].kwargs["is_first"])
        self.assertFalse(buffer.add.call_args_list[1].kwargs["is_first"])
        ai._player_number = True
        ai.round_end(SimpleNamespace(remaining_hps=[100, 90]))
        buffer.end_episode.assert_called_once()
        self.assertEqual(ai._outcomes[0].winner, "P1")

    def test_unknown_round_outcomes_rejected(self):
        ai = self.make_ai()
        ai._player_number = True
        bridge = fie._BridgeAI("test", queue.Queue(), queue.Queue(), "state", 16)
        for hps in (None, [], [10], [float("nan"), 10], [10, float("inf")]):
            for target in (ai, bridge):
                with self.subTest(hps=hps, target=type(target)), self.assertRaises(RuntimeError):
                    target.round_end(SimpleNamespace(remaining_hps=hps))


class PixelTests(unittest.TestCase):
    def test_missing_source_and_pixels_fail(self):
        for reader in (PixelReader(None), PixelReader(Mock(latest_pixels=lambda: None), timeout=0)):
            with self.assertRaises(RuntimeError):
                reader.read()
        with self.assertRaises(RuntimeError):
            fie._to_pixel_tensor(None, 16)

    def test_first_frame_wait_is_bounded(self):
        source = Mock(latest_pixels=Mock(return_value=None))
        with (
            patch("leworldgaming.env.match_contracts.time.monotonic", side_effect=[0, 0, 3]),
            patch("leworldgaming.env.match_contracts.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "No decoded"),
        ):
            PixelReader(source).read()
        sleep.assert_called_once_with(0.01)

    def test_waits_only_until_first_image(self):
        pixels = np.zeros((3, 16, 16), dtype=np.uint8)
        source = Mock(latest_pixels=Mock(side_effect=[None, pixels, pixels, None]))
        reader = PixelReader(source)
        with patch("leworldgaming.env.match_contracts.time.sleep") as sleep:
            self.assertIs(reader.read(), pixels)
            self.assertIs(reader.read(), pixels)
            with self.assertRaises(RuntimeError):
                reader.read()
        sleep.assert_called_once_with(0.01)

    def test_both_ai_observation_paths_require_pixels(self):
        targets = [
            ava._SelfDrivingAI("test", Mock(), "pixel", 16, None, []),
            fie._BridgeAI("test", queue.Queue(), queue.Queue(), "pixel", 16),
        ]
        for module, ai in zip((ava, fie), targets, strict=True):
            with (
                patch.object(module, "frame_to_obs_dict", return_value={}),
                self.assertRaisesRegex(RuntimeError, "spectator source"),
            ):
                ai._build_obs()


class EnvErrorTests(unittest.TestCase):
    def make_env(self):
        env = fie.FightingIceEnv()
        env._started = True
        return env

    def test_reset_and_step_raise_persistent_error(self):
        for method in ("reset", "step"):
            env = self.make_env()
            env._done = False
            failure = ValueError("controller failed")
            env._obs_q.put(("error", None, 0, True, {"error": failure}))
            with self.subTest(method=method), self.assertRaises(RuntimeError) as caught:
                env.reset() if method == "reset" else env.step(0)
            self.assertIs(caught.exception.__cause__, failure)
            self.assertTrue(env.match_over)
            with self.assertRaises(RuntimeError):
                env.reset()

    def test_thread_failure_surfaces_original_exception(self):
        env = self.make_env()
        failure = ValueError("thread failed")
        with (
            patch.object(env, "_match_main", AsyncMock(side_effect=failure)),
            self.assertLogs(fie.logger, level="ERROR"),
        ):
            env._thread_main()
        with self.assertRaises(RuntimeError) as caught:
            env.reset()
        self.assertIs(caught.exception.__cause__, failure)

    def test_normal_round_and_match_end_preserved(self):
        env = self.make_env()
        env._obs_q.put(("step", {"frame": 1}, 0, False, {"is_first": True}))
        self.assertEqual(env.reset()[0], {"frame": 1})
        env._obs_q.put(("round_end", {"frame": 1}, 0.5, True, {"win": True}))
        self.assertEqual(env.step(0)[1:4], (0.5, True, False))
        env._obs_q.put(("close", None, 0, True, {"match_over": True}))
        self.assertEqual(env.reset(), (None, {"match_over": True}))

    def test_bridge_close_does_not_publish_premature_success(self):
        env = self.make_env()
        bridge = fie._BridgeAI("test", env._obs_q, env._act_q, "state", 16)
        bridge.close()
        self.assertTrue(env._obs_q.empty())
        self.assertIsNone(env._act_q.get_nowait())

    def test_controller_failure_unblocks_parked_bridge_executor(self):
        env = fie.FightingIceEnv(fie.EnvConfig(opponent="random"))
        entered = threading.Event()

        def controller(_host, _port, ai, _is_p1):
            async def run():
                if isinstance(ai, fie._BridgeAI):
                    def processing():
                        entered.set()
                        env._act_q.get()
                    await asyncio.get_running_loop().run_in_executor(None, processing)
                else:
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                    raise ValueError("opponent failed")
            return SimpleNamespace(run=run)

        with ExitStack() as stack:
            network_mocks(stack, fie)
            stack.enter_context(patch.object(fie, "AIController", side_effect=controller))
            stack.enter_context(self.assertLogs(fie.logger, level="ERROR"))
            env._start_match()
            env._thread.join(timeout=2)
            try:
                self.assertFalse(env._thread.is_alive(), "executor did not shut down")
                with self.assertRaisesRegex(RuntimeError, "opponent failed"):
                    env.reset()
            finally:
                env.close()


class AsyncMatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_cancels_siblings_and_closes_connections(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def hanging():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def failing():
            await started.wait()
            raise ValueError("controller failed")

        for spectator_fails in (False, True):
            started.clear()
            cancelled.clear()
            hanging_ctrl = SimpleNamespace(run=hanging, writer=writer_mock())
            failing_ctrl = SimpleNamespace(run=failing, writer=writer_mock())
            with self.subTest(spectator_fails=spectator_fails), self.assertRaisesRegex(
                ValueError, "controller failed",
            ):
                await asyncio.wait_for(
                    run_controllers(
                        [hanging_ctrl] if spectator_fails else [hanging_ctrl, failing_ctrl],
                        failing_ctrl if spectator_fails else None,
                    ),
                    timeout=1,
                )
            self.assertTrue(cancelled.is_set())
            hanging_ctrl.writer.close.assert_called_once()
            failing_ctrl.writer.close.assert_called_once()

    async def test_normal_spectator_end_does_not_cancel_ai(self):
        finished = asyncio.Event()

        async def ai_run():
            await asyncio.sleep(0)
            finished.set()

        await run_controllers(
            [SimpleNamespace(run=ai_run)], SimpleNamespace(run=AsyncMock()),
        )
        self.assertTrue(finished.is_set())

    async def test_unexpected_controller_cancellation_is_failure(self):
        async def run():
            raise asyncio.CancelledError

        with self.assertRaisesRegex(RuntimeError, "Controller cancelled"):
            await run_controllers([SimpleNamespace(run=run)])

    async def test_external_cancellation_cleans_up_all_tasks(self):
        started = asyncio.Event()
        writer = writer_mock()

        async def run():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(run_controllers([SimpleNamespace(run=run, writer=writer)]))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        writer.close.assert_called_once()

    async def test_integer_jvm_rejection_cleans_up_both_entrypoints(self):
        for module in (ava, fie):
            with self.subTest(module=module.__name__), ExitStack() as stack:
                writer, gateway_close = network_mocks(stack, module, rejected=True)
                spectator = stack.enter_context(patch.object(module, "SpectatorRecorder")).return_value
                controller = stack.enter_context(patch.object(module, "AIController"))
                with self.assertRaisesRegex(RuntimeError, "JVM refused game: rejected"):
                    if module is ava:
                        await ava._run_match_async(Mock(), Mock(), p1_obs_mode="pixel")
                    else:
                        await fie.FightingIceEnv(fie.EnvConfig(obs_mode="pixel"))._match_main()
                writer.close.assert_called_once()
                writer.wait_closed.assert_awaited_once()
                gateway_close.assert_awaited_once()
                spectator.close.assert_called_once()
                controller.assert_not_called()

    async def test_empty_outcomes_rejected_by_both_entrypoints(self):
        for module in (ava, fie):
            with self.subTest(module=module.__name__), ExitStack() as stack:
                writer, _ = network_mocks(stack, module)
                stack.enter_context(patch.object(
                    module, "AIController", return_value=SimpleNamespace(run=AsyncMock()),
                ))
                with self.assertRaisesRegex(RuntimeError, "without valid round outcomes"):
                    if module is ava:
                        await ava._run_match_async(Mock(), Mock())
                    else:
                        env = fie.FightingIceEnv()
                        await env._match_main()
                if module is fie:
                    self.assertTrue(env._obs_q.empty())
                writer.close.assert_called_once()

    async def test_controller_errors_propagate_through_both_entrypoints(self):
        for module in (ava, fie):
            with self.subTest(module=module.__name__), ExitStack() as stack:
                writer, _ = network_mocks(stack, module)
                stack.enter_context(patch.object(
                    module, "AIController",
                    return_value=SimpleNamespace(run=AsyncMock(side_effect=ValueError("AI failed"))),
                ))
                with self.assertRaisesRegex(ValueError, "AI failed"):
                    if module is ava:
                        await ava._run_match_async(Mock(), Mock())
                    else:
                        await fie.FightingIceEnv()._match_main()
                writer.close.assert_called_once()

    async def test_custom_round_count_accepted_by_both_entrypoints(self):
        for module in (ava, fie):
            with self.subTest(module=module.__name__), ExitStack() as stack:
                network_mocks(stack, module)

                def controller(_host, _port, ai, is_p1):
                    async def run():
                        ai._player_number = is_p1
                        ai.round_end(SimpleNamespace(remaining_hps=[100, 80]))
                        ai.close()
                    return SimpleNamespace(run=run)

                stack.enter_context(patch.object(module, "AIController", side_effect=controller))
                if module is ava:
                    result = await ava._run_match_async(Mock(), Mock())
                    self.assertEqual(len(result.rounds), 1)
                    self.assertEqual(result.wins_p1, 1)
                    self.assertEqual(set(result.latency), {"AGENT_P1", "AGENT_P2"})
                else:
                    env = fie.FightingIceEnv()
                    await env._match_main()
                    self.assertEqual(env._bridge.completed_rounds, 1)
                    self.assertEqual(env._obs_q.get_nowait()[0], "round_end")
                    self.assertEqual(env._obs_q.get_nowait()[0], "close")


if __name__ == "__main__":
    unittest.main()
