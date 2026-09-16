"""Evaluation orchestration contracts, without a game server or trained weights."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from leworldgaming.eval.results import checkpoint_identity, write_result
from leworldgaming.eval.tournament import run_tournament, summarize, tournament_schedule

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("play", ROOT / "scripts/play.py")
play = importlib.util.module_from_spec(spec)
spec.loader.exec_module(play)


class EvaluationTests(unittest.TestCase):
    def test_seeded_schedule_is_side_balanced(self):
        schedule = tournament_schedule([0, 7])
        self.assertEqual(len(schedule), 12)
        for seed in (0, 7):
            pairs = {(m["p1"], m["p2"]) for m in schedule if m["seed"] == seed}
            self.assertEqual(len(pairs), 6)
            for p1, p2 in pairs:
                self.assertIn((p2, p1), pairs)
        for seeds in ([], [1, 1]):
            with self.assertRaises(ValueError):
                tournament_schedule(seeds)
        restricted = tournament_schedule([0], lewm_p1_only=True)
        self.assertEqual(len(restricted), 4)
        self.assertTrue(all(match["p2"] != "lewm" for match in restricted))

    def test_round_metrics_include_draws_and_correct_hp_orientation(self):
        record = {
            "arguments": {"p1": "lewm", "p2": "pets"},
            "result": {"rounds": [
                {"winner": "P1", "hp_p1": 100, "hp_p2": 0},
                {"winner": "draw", "hp_p1": 10, "hp_p2": 10},
            ]},
        }
        result = summarize([record])
        self.assertEqual(result["lewm_vs_pets"]["win_rate"], 0.5)
        self.assertEqual(result["lewm_vs_pets"]["mean_hp_diff"], 50)
        self.assertEqual(result["pets_vs_lewm"]["mean_hp_diff"], -50)
        self.assertEqual(result["pets_vs_lewm"]["draws"], 1)
        record["result"]["rounds"][0]["winner"] = None
        with self.assertRaises(ValueError):
            summarize([record])
        record["result"]["rounds"] = []
        with self.assertRaises(ValueError):
            summarize([record])

    def test_result_write_rejects_overwrite_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            write_result(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                write_result(path, {"ok": False})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            bad_path = Path(tmp) / "bad.json"
            with self.assertRaises(ValueError):
                write_result(bad_path, {"latency": float("nan")})
            self.assertFalse(bad_path.exists())

    def test_checkpoint_fingerprint_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            path.write_bytes(b"first")
            first = checkpoint_identity(path)
            path.write_bytes(b"other")
            self.assertNotEqual(first["sha256"], checkpoint_identity(path)["sha256"])

    @patch("leworldgaming.eval.tournament.dreamer_action_alignment", return_value="incoming_action_v1")
    @patch("leworldgaming.eval.tournament.code_identity", return_value={"revision": "test"})
    def test_successful_p1_fixed_tournament_persists_all_matches(self, _identity, _alignment):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.write_bytes(b"test")
            identity = checkpoint_identity(checkpoint)
            output = Path(tmp) / "results"

            def match(command, **_kwargs):
                def arg(name):
                    return command[command.index(name) + 1]

                write_result(arg("--output"), {
                    "arguments": {"p1": arg("--p1"), "p2": arg("--p2")},
                    "checkpoints": {"P1": identity, "P2": identity},
                    "code": {"revision": "test"},
                    "result": {"rounds": [{"winner": "P1", "hp_p1": 100, "hp_p2": 0}]},
                })

            with patch("leworldgaming.eval.tournament.subprocess.run", side_effect=match) as run:
                summary = run_tournament(
                    checkpoints=dict.fromkeys(("lewm", "dreamer", "pets"), str(checkpoint)),
                    output_dir=output, seeds=[3], games=1,
                    lewm_p1_only=True,
                )
            self.assertEqual(run.call_count, 4)
            self.assertEqual(summary["matches"], 4)
            self.assertEqual(len(summary["results"]), 6)
            self.assertEqual(summary["results"]["lewm_vs_dreamer"]["rounds"], 1)
            self.assertEqual(summary["results"]["lewm_vs_pets"]["rounds"], 1)
            for key in ("dreamer_vs_pets", "pets_vs_dreamer"):
                self.assertEqual(summary["results"][key]["rounds"], 2)
                self.assertEqual(summary["results"][key]["win_rate"], 0.5)
                self.assertEqual(summary["results"][key]["mean_hp_diff"], 0)
            for key in ("lewm_vs_dreamer", "lewm_vs_pets"):
                self.assertEqual(summary["results"][key]["win_rate"], 1.0)
                self.assertEqual(summary["results"][key]["mean_hp_diff"], 100)
            for key in ("dreamer_vs_lewm", "pets_vs_lewm"):
                self.assertEqual(summary["results"][key]["win_rate"], 0.0)
                self.assertEqual(summary["results"][key]["mean_hp_diff"], -100)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)

    @patch("leworldgaming.eval.tournament.dreamer_action_alignment", return_value="incoming_action_v1")
    @patch("leworldgaming.eval.tournament.code_identity", return_value={"revision": "test"})
    def test_dry_run_and_failed_match_never_write_final_summary(self, _identity, _alignment):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            path.write_bytes(b"test")
            kwargs = {
                "checkpoints": dict.fromkeys(("lewm", "dreamer", "pets"), str(path)),
                "output_dir": Path(tmp) / "results", "seeds": [1],
                "lewm_p1_only": True,
            }
            with patch("leworldgaming.eval.tournament.subprocess.run") as run:
                plan = run_tournament(**kwargs, dry_run=True)
                run.assert_not_called()
                self.assertEqual(len(plan["schedule"]), 4)
                self.assertFalse(kwargs["output_dir"].exists())
                run.side_effect = subprocess.CalledProcessError(1, "match")
                with self.assertRaises(subprocess.CalledProcessError):
                    run_tournament(**kwargs)
            self.assertTrue((kwargs["output_dir"] / "manifest.json").exists())
            self.assertFalse((kwargs["output_dir"] / "summary.json").exists())

    @patch("leworldgaming.eval.tournament.dreamer_action_alignment", return_value=None)
    @patch("leworldgaming.eval.tournament.code_identity", return_value={"revision": "test"})
    def test_legacy_and_perspective_blockers_prevent_game_launch(self, _identity, _alignment):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            path.write_bytes(b"test")
            kwargs = {
                "checkpoints": dict.fromkeys(("lewm", "dreamer", "pets"), str(path)),
                "output_dir": Path(tmp) / "results", "seeds": [0],
            }
            with patch("leworldgaming.eval.tournament.subprocess.run") as run:
                plan = run_tournament(**kwargs, dry_run=True, lewm_p1_only=True)
                self.assertEqual(len(plan["blockers"]), 1)
                with self.assertRaisesRegex(ValueError, "not cleared"):
                    run_tournament(**kwargs, lewm_p1_only=True)
                run.assert_not_called()
                self.assertFalse(kwargs["output_dir"].exists())

    @patch("leworldgaming.eval.tournament.dreamer_action_alignment", return_value="incoming_action_v1")
    @patch("leworldgaming.eval.tournament.code_identity", return_value={"revision": "test"})
    def test_p1_fixed_60hz_plan_uses_raw_lewm_blocks_and_pace(self, _identity, _alignment):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            path.write_bytes(b"test")
            identity = checkpoint_identity(path)
            output = Path(tmp) / "results"

            def match(command, **kwargs):
                self.assertEqual(kwargs["env"]["EVAL_PACE"], "realtime")
                self.assertIn("--pace", command)
                self.assertEqual(command[command.index("--pace") + 1], "realtime")
                if command[command.index("--p1") + 1] == "lewm":
                    self.assertEqual(command[command.index("--p1-frame-skip") + 1], "1")
                    self.assertEqual(command[command.index("--p1-planner") + 1], "cem")
                    self.assertIn("--planner-plan-raw-actions", command)
                    self.assertEqual(command[command.index("--planner-chunk-size") + 1], "1")
                write_result(command[command.index("--output") + 1], {
                    "arguments": {
                        "p1": command[command.index("--p1") + 1],
                        "p2": command[command.index("--p2") + 1],
                    },
                    "checkpoints": {"P1": identity, "P2": identity},
                    "code": {"revision": "test"},
                    "result": {"rounds": [{"winner": "P1", "hp_p1": 1, "hp_p2": 0}]},
                })

            with patch("leworldgaming.eval.tournament.subprocess.run", side_effect=match):
                result = run_tournament(
                    checkpoints=dict.fromkeys(("lewm", "dreamer", "pets"), str(path)),
                    output_dir=output, seeds=[0], games=1, pace="realtime", lewm_p1_only=True,
                )
            self.assertEqual(result["matches"], 4)

    def test_execution_cadence_matches_training_timestep(self):
        lewm = SimpleNamespace(temporal_stride=5, planner_plan_raw_actions=False)
        self.assertEqual(play.frame_skip_for("lewm", lewm, None), 5)
        with self.assertRaises(ValueError):
            play.frame_skip_for("lewm", lewm, 2)
        for name in ("pets", "dreamer"):
            self.assertEqual(play.frame_skip_for(name, object(), None), 1)
            with self.assertRaises(ValueError):
                play.frame_skip_for(name, object(), 8)
        lewm.planner_plan_raw_actions = True
        lewm.planner_name = "cem"
        lewm.planner_chunk_size = 1
        self.assertEqual(play.frame_skip_for("lewm", lewm, 1), 1)
        with self.assertRaises(ValueError):
            play.frame_skip_for("lewm", lewm, 5)
        self.assertEqual(play.frame_skip_for("random", object(), 5), 5)
        with self.assertRaises(ValueError):
            play.frame_skip_for("random", object(), 0)

    def test_trained_agents_require_checkpoints_before_construction(self):
        for name in ("lewm", "dreamer", "pets"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "requires"):
                play.build_agent(name, None, "cpu")

    def test_dreamer_legacy_checkpoint_requires_explicit_acknowledgement(self):
        agent = SimpleNamespace(action_alignment=None)
        with patch("leworldgaming.training.train_dreamer.build_agent_for_inference", return_value=agent):
            with self.assertRaisesRegex(ValueError, "legacy/unknown"):
                play.build_agent("dreamer", "legacy.pt", "cpu")
            with self.assertLogs(level="WARNING"):
                self.assertIs(
                    play.build_agent("dreamer", "legacy.pt", "cpu", allow_legacy_dreamer=True),
                    agent,
                )
            agent.action_alignment = "incoming_action_v1"
            self.assertIs(play.build_agent("dreamer", "corrected.pt", "cpu"), agent)

    def test_random_agent_is_seeded_and_only_commands_playable_actions(self):
        a = play.build_agent("random", None, "cpu", seed=42)
        b = play.build_agent("random", None, "cpu", seed=42)
        allowed = {action.to_int() for action in play.PLAYABLE_ACTIONS}
        sequence = [a.act({}) for _ in range(100)]
        self.assertEqual(sequence, [b.act({}) for _ in range(100)])
        self.assertTrue(set(sequence).issubset(allowed))


if __name__ == "__main__":
    unittest.main()
