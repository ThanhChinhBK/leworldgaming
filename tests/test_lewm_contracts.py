from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from torch import nn

from leworldgaming.agents.lewm.action_encoder import ActionEncoder
from leworldgaming.agents.lewm.planner import (
    _decode_pessimistic,
    _repeat_action_blocks,
    cem_shooting,
    random_shooting,
)
from leworldgaming.agents.lewm.twohot import make_bins, twohot_decode
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

    def test_decode_pessimistic_single_head_matches_plain_twohot_decode(self) -> None:
        bins = make_bins(5, -1.0, 1.0, "cpu")
        head = _ConstantLogitsHead(bins.numel(), peak_idx=2)
        z = torch.zeros(3, 4)
        out = _decode_pessimistic(head, bins, 0.5, z)
        expected = twohot_decode(head(z), bins)
        torch.testing.assert_close(out, expected)

    def test_decode_pessimistic_ensemble_applies_mean_minus_penalty_times_std(self) -> None:
        bins = make_bins(5, -1.0, 1.0, "cpu")
        heads = nn.ModuleList(
            [
                _ConstantLogitsHead(bins.numel(), peak_idx=0),
                _ConstantLogitsHead(bins.numel(), peak_idx=2),
                _ConstantLogitsHead(bins.numel(), peak_idx=4),
            ]
        )
        z = torch.zeros(3, 4)
        penalty = 0.7
        out = _decode_pessimistic(heads, bins, penalty, z)
        per_member = torch.stack([twohot_decode(member(z), bins) for member in heads], dim=0)
        expected = per_member.mean(dim=0) - penalty * per_member.std(dim=0)
        torch.testing.assert_close(out, expected)
        # Sanity: with real disagreement across members, penalizing must
        # pull the score below the plain mean (this is the whole point —
        # discourage the planner from trusting a lone optimistic member).
        plain_mean = per_member.mean(dim=0)
        self.assertTrue(torch.all(out < plain_mean))

    def test_decode_pessimistic_ensemble_zero_penalty_is_plain_mean(self) -> None:
        bins = make_bins(5, -1.0, 1.0, "cpu")
        heads = nn.ModuleList(
            [
                _ConstantLogitsHead(bins.numel(), peak_idx=0),
                _ConstantLogitsHead(bins.numel(), peak_idx=4),
            ]
        )
        z = torch.zeros(2, 4)
        out = _decode_pessimistic(heads, bins, 0.0, z)
        per_member = torch.stack([twohot_decode(member(z), bins) for member in heads], dim=0)
        torch.testing.assert_close(out, per_member.mean(dim=0))

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


class _ConstantLogitsHead(nn.Module):
    """Stub reward/value head: ignores its input, always emits a sharply
    peaked twohot logits vector at ``peak_idx`` (broadcasting over batch)."""

    def __init__(self, num_bins: int, peak_idx: int) -> None:
        super().__init__()
        logits = torch.full((num_bins,), -10.0)
        logits[peak_idx] = 10.0
        self.logits = logits

    def forward(self, z: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        return self.logits.unsqueeze(0).expand(z.shape[0], -1)


class PlannerSoftUpdateTests(unittest.TestCase):
    """#1 (MPPI soft update) + #2 (value_weight / reward_clip) knobs."""

    def _common(self):
        latent_dim = 8
        return dict(
            predictor=_IdentityPredictor(),
            pred_proj=nn.Identity(),
            action_encoder=ActionEncoder(
                action_dim=6 * 1, emb_dim=latent_dim, smoothed_dim=10
            ),
            probe=_FirstCoordinateProbe(),
            num_actions=6,
            horizon=2,
            num_samples=16,
            num_iters=2,
            history_size=3,
            temporal_stride=1,
        ), latent_dim

    def test_soft_update_runs_and_returns_valid_action(self) -> None:
        common, latent_dim = self._common()
        torch.manual_seed(0)
        action, dist = cem_shooting(
            torch.zeros(3, latent_dim), **common, elite_temp=1.0,
        )
        self.assertTrue(0 <= action < 6)
        # dist rows are valid probability distributions.
        self.assertEqual(dist.shape[-1], 6)
        torch.testing.assert_close(
            dist.sum(dim=-1), torch.ones(dist.shape[0]), atol=1e-4, rtol=0
        )

    def test_soft_update_favors_higher_scoring_action(self) -> None:
        # Probe returns first latent coord; _IdentityPredictor makes the
        # rollout track the encoded action, so distinct actions score
        # differently. A very low temperature (near-hard) elite refit should
        # still produce a normalized, peaked dist -- exercising the softmax
        # path without NaNs at small temp.
        common, latent_dim = self._common()
        torch.manual_seed(1)
        _, dist = cem_shooting(
            torch.zeros(2, latent_dim), **common, elite_temp=0.05,
        )
        self.assertTrue(torch.isfinite(dist).all())
        self.assertGreater(dist[0].max().item(), 0.0)

    def test_value_weight_and_reward_clip_accepted(self) -> None:
        common, latent_dim = self._common()
        torch.manual_seed(2)
        action, _ = cem_shooting(
            torch.zeros(2, latent_dim), **common,
            value_weight=0.5, reward_clip=1.0, elite_temp=0.0,
        )
        self.assertTrue(0 <= action < 6)

    def test_idle_penalty_suppresses_idle_action_in_dist(self) -> None:
        # With no reward/value heads (flat score everywhere), an idle
        # penalty on action id 0 should push probability mass at dist[0]
        # away from action 0 relative to no penalty at all.
        common, latent_dim = self._common()
        idle_ids = torch.tensor([0], dtype=torch.long)
        torch.manual_seed(3)
        _, dist_off = cem_shooting(
            torch.zeros(2, latent_dim), **common,
            idle_action_ids=idle_ids, idle_penalty=0.0,
        )
        torch.manual_seed(3)
        _, dist_on = cem_shooting(
            torch.zeros(2, latent_dim), **common,
            idle_action_ids=idle_ids, idle_penalty=5.0,
        )
        self.assertLess(dist_on[0, 0].item(), dist_off[0, 0].item())

    def test_idle_penalty_zero_is_noop(self) -> None:
        common, latent_dim = self._common()
        idle_ids = torch.tensor([0], dtype=torch.long)
        torch.manual_seed(4)
        action_a, dist_a = cem_shooting(
            torch.zeros(2, latent_dim), **common,
            idle_action_ids=idle_ids, idle_penalty=0.0,
        )
        torch.manual_seed(4)
        action_b, dist_b = cem_shooting(
            torch.zeros(2, latent_dim), **common,
            idle_action_ids=None, idle_penalty=0.0,
        )
        self.assertEqual(action_a, action_b)
        torch.testing.assert_close(dist_a, dist_b)

    def test_repeat_penalty_suppresses_repeat_of_prev_action(self) -> None:
        # With no reward/value heads (flat score everywhere), a repeat
        # penalty should push dist[0] away from prev_action relative to no
        # penalty at all -- the planner should prefer switching actions.
        common, latent_dim = self._common()
        common["num_samples"] = 512
        torch.manual_seed(6)
        _, dist_off = cem_shooting(
            torch.zeros(2, latent_dim), **common,
            repeat_penalty=0.0, prev_action=2,
        )
        torch.manual_seed(6)
        _, dist_on = cem_shooting(
            torch.zeros(2, latent_dim), **common,
            repeat_penalty=20.0, prev_action=2,
        )
        self.assertLess(dist_on[0, 2].item(), dist_off[0, 2].item())

    def test_repeat_penalty_zero_is_noop(self) -> None:
        common, latent_dim = self._common()
        torch.manual_seed(7)
        action_a, dist_a = cem_shooting(
            torch.zeros(2, latent_dim), **common,
            repeat_penalty=0.0, prev_action=None,
        )
        torch.manual_seed(7)
        action_b, dist_b = cem_shooting(
            torch.zeros(2, latent_dim), **common,
            repeat_penalty=0.0, prev_action=3,
        )
        self.assertEqual(action_a, action_b)
        torch.testing.assert_close(dist_a, dist_b)

    def test_policy_prior_shapes_initial_dist(self) -> None:
        # With no reward/value heads (flat score everywhere) and num_iters=1,
        # a peaked policy_prior should visibly bias the resulting elite-count
        # dist toward the prior's favored action relative to plain uniform
        # init, since CEM's first iteration samples straight from init_dist.
        common, latent_dim = self._common()
        prior = torch.zeros(6)
        prior[3] = 1.0
        torch.manual_seed(5)
        _, dist_uniform = cem_shooting(
            torch.zeros(2, latent_dim), **common, policy_prior=None,
        )
        torch.manual_seed(5)
        _, dist_prior = cem_shooting(
            torch.zeros(2, latent_dim), **common, policy_prior=prior,
        )
        self.assertGreater(dist_prior[0, 3].item(), dist_uniform[0, 3].item())

    def test_policy_prior_none_is_noop(self) -> None:
        common, latent_dim = self._common()
        torch.manual_seed(6)
        action_a, dist_a = cem_shooting(
            torch.zeros(2, latent_dim), **common, policy_prior=None,
        )
        torch.manual_seed(6)
        action_b, dist_b = cem_shooting(
            torch.zeros(2, latent_dim), **common,
        )
        self.assertEqual(action_a, action_b)
        torch.testing.assert_close(dist_a, dist_b)

    def test_policy_prior_masked_to_valid_actions(self) -> None:
        # A prior that puts all its mass on an invalid action must fall back
        # to uniform-over-valid rather than silently biasing toward an
        # action CEM isn't allowed to sample.
        common, latent_dim = self._common()
        valid = torch.tensor([0, 1, 2], dtype=torch.long)
        prior = torch.zeros(6)
        prior[5] = 1.0  # outside `valid`
        torch.manual_seed(7)
        action, dist = cem_shooting(
            torch.zeros(2, latent_dim),
            **{**common, "valid_actions": valid},
            policy_prior=prior,
        )
        self.assertIn(action, (0, 1, 2))
        self.assertAlmostEqual(dist[0, 3].item(), 0.0, places=5)
        self.assertAlmostEqual(dist[0, 4].item(), 0.0, places=5)
        self.assertAlmostEqual(dist[0, 5].item(), 0.0, places=5)


class OnlineOpponentModelTests(unittest.TestCase):
    def _obs(self, attacking: bool, hp: float = 400.0) -> dict:
        return {
            "own": {"x": 100, "y": 0, "hp": hp, "energy": 50},
            "opp": {
                "x": 130, "y": 0, "hp": 300, "energy": 50,
                "action": 35 if attacking else 1,
                "atk_is_live": 1 if attacking else 0,
                "atk_start_up": 3 if attacking else 0,
                "control": 1,
            },
            "global": {"max_hp": 400, "proj_opp": 0},
        }

    def test_online_fit_separates_threat(self) -> None:
        from leworldgaming.agents.lewm.online_opponent_model import (
            OnlineOpponentModel,
        )
        om = OnlineOpponentModel(lr=0.02)
        hp = 400.0
        for t in range(3000):
            attacking = (t % 3 == 0)
            obs = self._obs(attacking, hp)
            om.observe_outcome(obs)
            om.predict_threat(obs)
            if attacking:
                hp -= 8.0
                if hp <= 4.0:
                    hp = 400.0
        p_attack = om.predict_threat(self._obs(True, hp))
        p_idle = om.predict_threat(self._obs(False, hp))
        # After online training the threat estimate should clearly separate
        # an attacking opponent from an idle one.
        self.assertGreater(p_attack, 0.7)
        self.assertLess(p_idle, 0.3)

    def test_bias_direction_and_normalization(self) -> None:
        import numpy as np
        from leworldgaming.agents.lewm.online_opponent_model import (
            OnlineOpponentModel,
            _ATTACK_IDS,
            _EVASION_IDS,
            _GUARD_IDS,
        )
        om = OnlineOpponentModel(strength=1.5)
        dist = np.ones(56) / 56.0
        threatened = om.bias_action_dist(dist.copy(), 0.95)
        safe = om.bias_action_dist(dist.copy(), 0.05)
        # Distributions stay normalized.
        self.assertAlmostEqual(float(threatened.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(safe.sum()), 1.0, places=5)
        defend_ids = list(_GUARD_IDS + _EVASION_IDS)
        attack_ids = list(_ATTACK_IDS)
        # Under threat, defensive mass rises vs uniform and attack mass falls.
        self.assertGreater(threatened[defend_ids].sum(), dist[defend_ids].sum())
        self.assertLess(threatened[attack_ids].sum(), dist[attack_ids].sum())
        # When safe, attack mass rises vs uniform.
        self.assertGreater(safe[attack_ids].sum(), dist[attack_ids].sum())

    def test_neutral_threat_is_noop(self) -> None:
        import numpy as np
        from leworldgaming.agents.lewm.online_opponent_model import (
            OnlineOpponentModel,
        )
        om = OnlineOpponentModel(strength=1.5)
        dist = np.ones(56) / 56.0
        out = om.bias_action_dist(dist.copy(), 0.5)  # == threshold
        np.testing.assert_allclose(out, dist)

    def test_missing_keys_never_crash(self) -> None:
        from leworldgaming.agents.lewm.online_opponent_model import (
            OnlineOpponentModel,
        )
        om = OnlineOpponentModel()
        # Empty / partial obs should not raise.
        om.observe_outcome({})
        p = om.predict_threat({})
        self.assertTrue(0.0 <= p <= 1.0)


if __name__ == "__main__":
    unittest.main()
