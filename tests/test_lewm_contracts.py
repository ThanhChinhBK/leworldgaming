from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from torch import nn

from leworldgaming.agents.lewm.action_encoder import ActionEncoder
from leworldgaming.agents.lewm.planner import _repeat_action_blocks, random_shooting
from leworldgaming.data.replay_buffer import DataReader, _MultiStarts
from leworldgaming.training._replay_utils import reduce_reward_seq, to_device_seq
from leworldgaming.training.train_lewm_heads import _lambda_return
from leworldgaming.utils.image import IMAGENET_MEAN, IMAGENET_STD, normalize_imagenet_pixels


class LeWMContractTests(unittest.TestCase):
    def test_reward_alignment_stride_one(self) -> None:
        raw = torch.tensor([[100.0, 1.0, 2.0, 3.0]])
        actual = reduce_reward_seq(raw, stride=1)
        torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_stride_one_drops_endpoint_action(self) -> None:
        pixels = np.zeros((1, 4, 3, 2, 2), dtype=np.uint8)
        actions = np.array([[0, 1, 2, 3]], dtype=np.int64)
        _, action_blocks = to_device_seq(
            pixels, actions, action_dim=4, device=torch.device("cpu"), stride=1
        )
        self.assertEqual(action_blocks.shape, (1, 3, 4))

    def test_reward_alignment_stride_five(self) -> None:
        raw = torch.arange(11, dtype=torch.float32).unsqueeze(0)
        actual = reduce_reward_seq(raw, stride=5)
        torch.testing.assert_close(
            actual,
            torch.tensor([[1 + 2 + 3 + 4 + 5, 6 + 7 + 8 + 9 + 10.0]]),
        )

    def test_episode_split_has_no_episode_overlap(self) -> None:
        starts = _MultiStarts(
            file_indices=np.array([0, 0, 0, 0, 0, 0, 1, 1]),
            local_starts=np.arange(8),
            episode_indices=np.array([0, 0, 1, 1, 2, 2, 0, 1]),
        )
        train, val = starts.split_by_episode(0.25, seed=7)
        train_episodes = set(
            zip(train.file_indices, train.episode_indices, strict=True)
        )
        val_episodes = set(
            zip(val.file_indices, val.episode_indices, strict=True)
        )
        self.assertTrue(train_episodes)
        self.assertTrue(val_episodes)
        self.assertTrue(train_episodes.isdisjoint(val_episodes))

    def test_reader_rejects_active_writer(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.h5"
            with h5py.File(path, "w") as replay:
                replay.create_dataset("action", data=np.array([0]))
                replay.create_dataset("reward", data=np.array([0.0]))
                replay.create_dataset("done", data=np.array([1], dtype=np.uint8))
                replay.create_dataset("episode_starts", data=np.array([0]))
                replay.attrs["writer_active"] = np.uint8(1)
            with self.assertRaisesRegex(RuntimeError, "still open for writing"):
                DataReader(path)

    def test_reader_rejects_interleaved_perspectives(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "swapped.h5"
            n = 100
            own = np.tile(np.array([400, 350], dtype=np.int32), n // 2)
            opp = np.tile(np.array([350, 400], dtype=np.int32), n // 2)
            with h5py.File(path, "w") as replay:
                replay.create_dataset("action", data=np.zeros(n, dtype=np.int16))
                replay.create_dataset("reward", data=np.zeros(n, dtype=np.float32))
                done = np.zeros(n, dtype=np.uint8)
                done[-1] = 1
                replay.create_dataset("done", data=done)
                replay.create_dataset("episode_starts", data=np.array([0]))
                replay.create_dataset("obs/own/hp", data=own)
                replay.create_dataset("obs/opp/hp", data=opp)
            with self.assertRaisesRegex(RuntimeError, "interleaved player perspectives"):
                DataReader(path)

    def test_lambda_return_keeps_final_reward_and_bootstrap(self) -> None:
        rewards = torch.tensor([[1.0, 2.0]])
        cont = torch.ones_like(rewards)
        next_values = torch.tensor([[10.0, 20.0]])
        td_zero = _lambda_return(rewards, cont, next_values, gamma=1.0, lam=0.0)
        monte_carlo = _lambda_return(rewards, cont, next_values, gamma=1.0, lam=1.0)
        torch.testing.assert_close(td_zero, torch.tensor([[11.0, 22.0]]))
        torch.testing.assert_close(monte_carlo, torch.tensor([[23.0, 22.0]]))

    def test_terminal_stops_lambda_bootstrap(self) -> None:
        rewards = torch.tensor([[1.0, 2.0]])
        cont = torch.tensor([[1.0, 0.0]])
        next_values = torch.tensor([[10.0, 20.0]])
        actual = _lambda_return(rewards, cont, next_values, gamma=1.0, lam=1.0)
        torch.testing.assert_close(actual, torch.tensor([[3.0, 2.0]]))

    def test_action_block_repeats_high_level_action_for_stride(self) -> None:
        actions = torch.tensor([[2, 1]])
        blocks = _repeat_action_blocks(actions, num_actions=4, temporal_stride=3)
        self.assertEqual(blocks.shape, (1, 2, 12))
        expected = torch.nn.functional.one_hot(actions, num_classes=4).float()
        expected = expected.unsqueeze(-2).expand(1, 2, 3, 4).reshape(1, 2, 12)
        torch.testing.assert_close(blocks, expected)

    def test_stride_five_planner_accepts_effective_action_width(self) -> None:
        latent_dim = 8
        predictor = _IdentityPredictor()
        action_encoder = ActionEncoder(
            action_dim=56 * 5, emb_dim=latent_dim, smoothed_dim=10
        )
        action = random_shooting(
            torch.zeros(3, latent_dim),
            predictor,
            nn.Identity(),
            action_encoder,
            _FirstCoordinateProbe(),
            num_actions=56,
            horizon=2,
            num_samples=4,
            history_size=3,
            temporal_stride=5,
        )
        self.assertTrue(0 <= action < 56)

    def test_imagenet_normalization_matches_reference_constants(self) -> None:
        pixels = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
        normalized = normalize_imagenet_pixels(pixels)
        expected = torch.tensor(
            [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)]
        )
        torch.testing.assert_close(normalized[0, :, 0, 0], expected)


class _IdentityPredictor(nn.Module):
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        self.last_condition_shape = c.shape
        return x


class _FirstCoordinateProbe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


if __name__ == "__main__":
    unittest.main()
