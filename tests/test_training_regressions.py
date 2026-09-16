from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from leworldgaming.data.dreamer_export import (
    ACTION_ALIGNMENT,
    _aligned_episode,
    export_episodes_to_npz,
)
from leworldgaming.data.replay_buffer import _MultiStarts
from leworldgaming.training.train_dreamer import (
    _build_dreamer_config,
    _require_action_alignment,
)
from leworldgaming.training.train_dreamer import _load_config as load_dreamer_config
from leworldgaming.training.train_pets import _load_config as load_pets_config
from leworldgaming.training.train_pets import _snapshot_training_state, _validate_resume_progress


class TrainingRegressionTests(unittest.TestCase):
    def test_playable_mask_setting_survives_training_config(self) -> None:
        for enabled in (True, False):
            overrides = {"restrict_to_playable_actions": enabled}
            dreamer = load_dreamer_config(None, overrides)
            pets = load_pets_config(None, overrides)
            self.assertIs(dreamer["restrict_to_playable_actions"], enabled)
            self.assertIs(pets["restrict_to_playable_actions"], enabled)
            namespace = _build_dreamer_config(dreamer, torch.device("cpu"))
            self.assertIs(namespace.restrict_to_playable_actions, enabled)

    def test_dreamer_incoming_actions_leave_rewards_unchanged(self) -> None:
        actions = np.array([2, 1, 3, 0])
        rewards = np.array([0.0, 0.25, -0.1, 0.5], dtype=np.float32)
        picks, episode = _aligned_episode(actions, rewards, np.array([0, 0, 0, 1]), 4, 1)
        np.testing.assert_array_equal(picks, [0, 1, 2, 3])
        np.testing.assert_array_equal(episode["action"][0], np.zeros(4))
        np.testing.assert_array_equal(episode["action"][1:].argmax(-1), actions[:-1])
        np.testing.assert_array_equal(episode["reward"], rewards)
        np.testing.assert_array_equal(episode["is_terminal"], [False, False, False, True])

    def test_strided_episode_keeps_endpoint_and_incoming_reward(self) -> None:
        actions = np.array([1, 1, 2, 2, 3, 3])
        rewards = np.arange(6, dtype=np.float32)
        picks, episode = _aligned_episode(actions, rewards, np.array([0, 0, 0, 0, 0, 1]), 4, 2)
        np.testing.assert_array_equal(picks, [0, 2, 4, 5])
        np.testing.assert_array_equal(episode["action"][0], np.zeros(4))
        np.testing.assert_array_equal(episode["action"][1:].argmax(-1), [1, 2, 3])
        np.testing.assert_array_equal(episode["reward"], [0, 3, 7, 5])
        self.assertEqual(episode["reward"].sum(), rewards.sum())
        np.testing.assert_array_equal(episode["is_terminal"], [False, False, False, True])

    def test_nonterminal_endpoint_not_forced_terminal(self) -> None:
        _, episode = _aligned_episode(np.array([1, 2]), np.zeros(2), np.zeros(2), 3, 5)
        self.assertFalse(episode["is_terminal"].any())

    def test_invalid_stride_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stride"):
            _aligned_episode(np.array([1, 2]), np.zeros(2), np.zeros(2), 3, 0)

    def test_legacy_dreamer_cannot_be_relabelled_by_resume(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy/misaligned"):
            _require_action_alignment({"num_steps": 71000})
        _require_action_alignment({"action_alignment": ACTION_ALIGNMENT})

    def test_legacy_cache_refused_without_deleting_files(self) -> None:
        with (
            patch("leworldgaming.data.dreamer_export._resolve_h5_paths", return_value=[]),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value="vector_v1"),
            patch.object(Path, "glob", return_value=[Path("existing.npz")]),
            patch.object(Path, "unlink") as unlink,
        ):
            with self.assertRaisesRegex(ValueError, "fresh episode_dir"):
                export_episodes_to_npz("source.h5", "existing-cache")
            unlink.assert_not_called()
        self.assertEqual(ACTION_ALIGNMENT, "incoming_action_v1")

    def test_episode_partition_has_no_shared_episode(self) -> None:
        starts = _MultiStarts(
            np.zeros(12, dtype=np.int32),
            np.arange(12),
            np.repeat(np.arange(4), 3),
        )
        train, val = starts.split_by_episode(0.25, 0)
        self.assertFalse(set(train.episode_indices) & set(val.episode_indices))
        with self.assertRaisesRegex(ValueError, "at least two episodes"):
            starts[:3].split_by_episode(0.25, 0)

    def test_best_pets_snapshot_keeps_matching_optimizer_and_progress(self) -> None:
        class Optimizer:
            def __init__(self) -> None:
                self.state = {"state": {0: {"step": torch.tensor(7.0), "exp_avg": torch.tensor([2.0])}}}

            def state_dict(self) -> dict:
                return self.state

        dynamics = torch.nn.Linear(1, 1, bias=False)
        optim = Optimizer()
        snapshot = _snapshot_training_state(dynamics, optim, 7)
        selected_weight = snapshot["dynamics"]["weight"].clone()
        with torch.no_grad():
            dynamics.weight.add_(1)
        optim.state["state"][0]["step"].add_(3)
        optim.state["state"][0]["exp_avg"].zero_()
        torch.testing.assert_close(snapshot["dynamics"]["weight"], selected_weight)
        self.assertEqual(snapshot["num_steps"], 7)
        self.assertEqual(snapshot["optim"]["state"][0]["step"].item(), 7)
        self.assertEqual(snapshot["optim"]["state"][0]["exp_avg"].item(), 2)
        _validate_resume_progress(snapshot)
        snapshot["num_steps"] = 10
        with self.assertRaisesRegex(ValueError, "optimizer progress"):
            _validate_resume_progress(snapshot)


if __name__ == "__main__":
    unittest.main()
