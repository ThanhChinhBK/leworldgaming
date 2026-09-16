from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from leworldgaming.agents.dreamer.agent import DreamerAgent
from leworldgaming.agents.lewm.agent import LewmAgent
from leworldgaming.agents.lewm.continuation_head import ContinuationHead
from leworldgaming.agents.lewm.mcts_planner import mcts_search
from leworldgaming.agents.lewm.online_opponent_model import _EVASION_IDS
from leworldgaming.agents.lewm.planner import _prepare_context, _score_action_sequences
from leworldgaming.agents.pets.agent import PETSAgent
from leworldgaming.agents.pets.cem_planner import CEMPlannerDiscrete
from leworldgaming.env.action_space import commandable_action_ids, mask_action_logits


def tiny_config(**overrides):
    return {
        "latent_dim": 8, "action_dim": 6, "history_size": 3,
        "encoder_image_size": 8, "encoder_patch_size": 4,
        "encoder_embed_dim": 8, "encoder_depth": 1, "encoder_heads": 2,
        "projector_hidden": 16, "predictor_depth": 1, "predictor_heads": 2,
        "predictor_dim_head": 4, "predictor_mlp_dim": 16,
        "heads": {"hidden_dim": 8},
        **overrides,
    }


class _FrameEncoder(nn.Module):
    def forward(self, pixels):
        return pixels[:, 0, 0, 0, None].expand(-1, 8)


class _CaptureActions(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = []

    def forward(self, blocks):
        self.blocks.append(blocks.clone())
        return blocks.sum(dim=-1, keepdim=True).expand(*blocks.shape[:-1], 2)


class _CheckedPredictor(nn.Module):
    def forward(self, z, actions):
        if z.shape != actions.shape:
            raise ValueError(f"History mismatch: {z.shape} != {actions.shape}")
        return z + 1


class _Probe(nn.Module):
    def forward(self, z):
        return z[:, :1]


class _OneHotDistribution(torch.distributions.OneHotCategorical):
    def mode(self):
        onehot = nn.functional.one_hot(self.logits.argmax(-1), self.logits.shape[-1])
        return onehot.detach() + self.logits - self.logits.detach()


class _FixedActor(nn.Module):
    def __init__(self):
        super().__init__()
        logits = torch.zeros(1, 56)
        logits[0, 1] = 10.0
        logits[0, 2] = 9.0
        self.register_buffer("logits", logits)

    def forward(self, features):
        return _OneHotDistribution(logits=self.logits)


class _StubDreamer(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = _FixedActor()
        self._task_behavior = SimpleNamespace(actor=self.actor)
        self.received_state = None

    def forward(self, batch, reset, state, training=False):
        self.received_state = state
        action = self.actor(torch.zeros(1, 2)).mode()
        return {"action": action}, (torch.zeros(1, 2), action)


class _ActionDynamics:
    ensemble_size = 1

    def __init__(self):
        self.actions = []

    def predict(self, state, actions, members, sample=True):
        self.actions.extend(actions.tolist())
        return actions.float().unsqueeze(-1)


class ModelRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_ensemble_checkpoint_roundtrip(self):
        agent = LewmAgent(tiny_config(heads={
            "hidden_dim": 8, "reward_ensemble_size": 2, "value_ensemble_size": 2,
        }))
        expected = {
            name: {k: v.clone() for k, v in getattr(agent, name).state_dict().items()}
            for name in ("reward_head", "value_head")
        }
        checkpoint = io.BytesIO()
        agent.save(checkpoint)
        checkpoint.seek(0)
        agent.load(checkpoint)
        for name, state in expected.items():
            for key, value in state.items():
                torch.testing.assert_close(getattr(agent, name).state_dict()[key], value)

    def test_legacy_modulelist_checkpoint_loads(self):
        heads = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        restored = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        LewmAgent._load_head(restored, {"reward_head": heads.state_dict()},
                             "reward_heads", "reward_head")
        for key, value in heads.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_policy_checkpoint_roundtrip(self):
        agent = LewmAgent(tiny_config(heads={
            "hidden_dim": 8, "policy_loss_weight": 1.0, "policy_hidden_dim": 8,
        }))
        expected = {k: v.clone() for k, v in agent.policy_head.state_dict().items()}
        checkpoint = io.BytesIO()
        agent.save(checkpoint)
        checkpoint.seek(0)
        agent.load(checkpoint)
        for key, value in expected.items():
            torch.testing.assert_close(agent.policy_head.state_dict()[key], value)

    def test_missing_enabled_policy_weights_fail_explicitly(self):
        agent = LewmAgent(tiny_config(heads={
            "hidden_dim": 8, "policy_loss_weight": 1.0, "policy_hidden_dim": 8,
        }))
        checkpoint = io.BytesIO()
        agent.save(checkpoint)
        checkpoint.seek(0)
        state = torch.load(checkpoint, weights_only=False)
        del state["policy_head"]
        with patch("torch.load", return_value=state), self.assertRaisesRegex(
            ValueError, "no policy_head weights"
        ):
            agent.load("unused")

    def test_chunk_warm_shift_counts_initial_action(self):
        agent = LewmAgent(tiny_config(planner={"chunk_size": 3, "horizon": 4}))
        shifts = []

        def plan(*args, **kwargs):
            shifts.append(kwargs["warm_shift"])
            return 0, torch.ones(4, 6) / 6

        with patch("leworldgaming.agents.lewm.agent.cem_shooting", plan):
            for _ in range(7):
                agent.act({"pixels": torch.zeros(3, 8, 8)})
        self.assertEqual(shifts, [1, 3, 3])

    def test_raw_agent_context_tracks_stride_and_complete_blocks(self):
        for chunk_size in (1, 2):
            with self.subTest(chunk_size=chunk_size):
                agent = LewmAgent(tiny_config(
                    temporal_stride=2,
                    planner={"chunk_size": chunk_size, "plan_raw_actions": True},
                ))
                agent.encoder = _FrameEncoder()
                agent.projector = nn.Identity()
                records = {}

                def plan(z, records=records, **kwargs):
                    t = int(z[-1, 0])
                    records[t] = (z[:, 0].tolist(), kwargs["past_actions"].tolist())
                    dist = torch.zeros(4, 6)
                    for k in range(4):
                        dist[k, (t + k) % 6] = 1
                    return t % 6, dist

                with patch("leworldgaming.agents.lewm.agent.cem_shooting", plan):
                    for t in range(5):
                        agent.act({"pixels": torch.full((3, 8, 8), float(t))})
                self.assertEqual(records[4], ([0, 2, 4], [[0, 1], [2, 3]]))
                if chunk_size == 1:
                    self.assertEqual(records[3], ([1, 3], [[1, 2]]))
                agent.reset_episode()
                self.assertEqual(agent._z_history, [])
                self.assertEqual(agent._action_history, [])

    def test_switching_raw_mode_clears_incompatible_history(self):
        agent = LewmAgent(tiny_config())
        agent._z_history = [torch.zeros(8)]
        agent._action_history = [1]
        agent.configure_planner(plan_raw_actions=True)
        self.assertEqual(agent._z_history, [])
        self.assertEqual(agent._action_history, [])

    def test_raw_agent_runs_full_cem_across_replans(self):
        for history_size in (1, 3):
            with self.subTest(history_size=history_size):
                agent = LewmAgent(tiny_config(
                    temporal_stride=2, history_size=history_size,
                    planner={
                        "chunk_size": 2, "plan_raw_actions": True,
                        "horizon": 2, "num_samples": 4, "num_iters": 1,
                    },
                ))
                for _ in range(4):
                    action = agent.act({"pixels": torch.zeros(3, 8, 8)})
                    self.assertIn(action, range(6))

    def test_prepare_context_pads_complete_raw_blocks(self):
        _, history = _prepare_context(
            torch.zeros(2, 2), torch.tensor([[1, 2]]), 2, 3, torch.device("cpu")
        )
        self.assertEqual(history.tolist(), [[[0, 0], [1, 2]], [[0, 0], [1, 2]]])

    def test_raw_rollout_retains_full_previous_block(self):
        encoder = _CaptureActions()
        _score_action_sequences(
            torch.zeros(1, 2, 2), torch.tensor([[0]]), None,
            _CheckedPredictor(), nn.Identity(), encoder, _Probe(),
            3, 2, None, None, None, None, None, 1.0,
            sub_actions=torch.tensor([[[1, 2], [0, 1]]]),
        )
        previous_block = encoder.blocks[1][0, 0].reshape(2, 3).argmax(-1)
        self.assertEqual(previous_block.tolist(), [1, 2])

    def test_raw_rollout_history_size_one_stays_empty(self):
        encoder = _CaptureActions()
        _score_action_sequences(
            torch.zeros(1, 1, 2), torch.empty(1, 0, dtype=torch.long), None,
            _CheckedPredictor(), nn.Identity(), encoder, _Probe(),
            3, 2, None, None, None, None, None, 1.0,
            sub_actions=torch.tensor([[[1, 2], [0, 1]]]),
        )
        self.assertTrue(all(block.shape == (1, 1, 6) for block in encoder.blocks))

    def test_mcts_singleton_continuation_preserves_batch_axis(self):
        action, distribution = mcts_search(
            torch.zeros(2, 2), _CheckedPredictor(), nn.Identity(),
            _CaptureActions(), _Probe(), 2,
            num_simulations=1, sim_batch_size=1, history_size=2,
            continuation_head=ContinuationHead(2, 4), dirichlet_frac=0,
        )
        self.assertIn(action, (0, 1))
        torch.testing.assert_close(distribution.sum(), torch.tensor(1.0))

    def test_mcts_history_size_one_can_expand_deeper(self):
        encoder = _CaptureActions()
        action, _ = mcts_search(
            torch.zeros(1, 2), _CheckedPredictor(), nn.Identity(),
            encoder, _Probe(), 1,
            num_simulations=3, sim_batch_size=1, history_size=1,
            max_depth=3, dirichlet_frac=0,
        )
        self.assertEqual(action, 0)
        self.assertEqual(len(encoder.blocks), 3)
        self.assertTrue(all(block.shape[1] == 1 for block in encoder.blocks))

    def test_evasion_ids_match_named_actions(self):
        from pyftg.models.enums.action import Action

        self.assertEqual(
            {Action.from_int(i).name for i in _EVASION_IDS},
            {"BACK_STEP", "BACK_JUMP", "JUMP"},
        )

    def test_shared_mask_normalizes_without_invalid_action_mass(self):
        logits = torch.zeros(3, 56)
        logits[:, 1] = 100
        logits[:, 2] = 3
        masked = mask_action_logits(logits)
        probs = torch.softmax(masked, -1)
        valid = commandable_action_ids()
        invalid = [i for i in range(56) if i not in valid]
        self.assertEqual(len(valid), 42)
        self.assertEqual(probs[:, invalid].count_nonzero().item(), 0)
        self.assertEqual(masked.argmax(-1).tolist(), [2, 2, 2])
        torch.testing.assert_close(probs.sum(-1), torch.ones(3))
        torch.testing.assert_close(mask_action_logits(logits, False), logits)
        self.assertIsNone(commandable_action_ids(6))

    def test_pets_masks_sampling_refitting_and_final_selection(self):
        torch.manual_seed(0)
        dynamics = _ActionDynamics()
        distributions = []
        categorical = torch.distributions.Categorical

        def capture(**kwargs):
            distribution = categorical(**kwargs)
            distributions.append(distribution.probs.clone())
            return distribution

        planner = CEMPlannerDiscrete(
            56, horizon=1, num_candidates=128, num_elites=1, num_iters=3,
        )
        with patch("torch.distributions.Categorical", capture):
            action = planner.plan(
                torch.zeros(1), dynamics,
                reward_fn=lambda s, nxt: 100 * (nxt[:, 0] == 1) + 10 * (nxt[:, 0] == 2),
            )
        self.assertEqual(action, 2)
        self.assertTrue(set(dynamics.actions).issubset(commandable_action_ids()))
        for probs in distributions:
            self.assertEqual(probs[0, 1].item(), 0)
            torch.testing.assert_close(probs.sum(-1), torch.ones(1))

    def test_mask_handles_finite_minimum_logits(self):
        logits = torch.full((1, 56), torch.finfo(torch.float32).min)
        probs = mask_action_logits(logits).softmax(-1)
        self.assertEqual(probs[0, 1].item(), 0)
        torch.testing.assert_close(probs.sum(-1), torch.ones(1))

    def test_mask_preserves_upstream_dreamer_straight_through_mode(self):
        from tools import OneHotDist

        logits = torch.zeros(1, 56)
        logits[0, 1] = 100
        logits[0, 2] = 3
        dist = DreamerAgent._mask_actor_distribution(None, None, OneHotDist(logits=logits))
        self.assertTrue(torch.isfinite(dist.mode()).all())
        self.assertEqual(dist.mode().argmax(-1).item(), 2)
        self.assertEqual(dist.probs[0, 1].item(), 0)
        torch.testing.assert_close(dist.probs.sum(-1), torch.ones(1))

    def test_pets_legacy_opt_out_can_select_stand(self):
        torch.manual_seed(0)
        planner = CEMPlannerDiscrete(
            56, horizon=1, num_candidates=500, num_elites=1, num_iters=2,
            restrict_to_playable_actions=False,
        )
        action = planner.plan(
            torch.zeros(1), _ActionDynamics(),
            reward_fn=lambda s, nxt: (nxt[:, 0] == 1).float(),
        )
        self.assertEqual(action, 1)

    def test_pets_checkpoint_preserves_explicit_mask_opt_out(self):
        agent = PETSAgent({
            "state_dim": 2, "hidden": 4, "num_layers": 1,
            "ensemble_size": 1, "action_emb_dim": 2,
        })
        agent.restrict_to_playable_actions = False
        checkpoint = io.BytesIO()
        agent.save(checkpoint)
        checkpoint.seek(0)
        agent.load(checkpoint)
        self.assertFalse(agent.restrict_to_playable_actions)
        self.assertFalse(agent.planner.restrict_to_playable_actions)

    def test_dreamer_masks_before_mode_and_carries_selected_action(self):
        module = _StubDreamer()
        agent = DreamerAgent(module, SimpleNamespace(), torch.device("cpu"))
        obs = {"vector": torch.zeros(42).numpy()}
        self.assertEqual(agent.act(obs), 2)
        self.assertTrue(torch.isfinite(agent._state[1]).all())
        self.assertEqual(agent._state[1].argmax(-1).item(), 2)
        self.assertEqual(agent.act(obs), 2)
        self.assertEqual(module.received_state[1].argmax(-1).item(), 2)
        self.assertEqual(len(module.actor._forward_hooks), 0)

    def test_dreamer_legacy_opt_out_can_select_stand(self):
        agent = DreamerAgent(
            _StubDreamer(), SimpleNamespace(restrict_to_playable_actions=False),
            torch.device("cpu"),
        )
        self.assertEqual(agent.act({"vector": torch.zeros(42).numpy()}), 1)

    def test_dreamer_roundtrip_preserves_alignment_without_blessing_legacy(self):
        agent = DreamerAgent(_StubDreamer(), SimpleNamespace(), torch.device("cpu"))
        agent.action_alignment = "incoming_action_v1"
        agent.restrict_to_playable_actions = False
        checkpoint = io.BytesIO()
        agent.save(checkpoint)
        checkpoint.seek(0)
        agent.load(checkpoint)
        self.assertEqual(agent.action_alignment, "incoming_action_v1")
        self.assertFalse(agent.restrict_to_playable_actions)
        checkpoint.seek(0)
        legacy = torch.load(checkpoint, weights_only=False)
        del legacy["action_alignment"]
        with patch("torch.load", return_value=legacy):
            agent.load("unused")
        self.assertIsNone(agent.action_alignment)
        self.assertIsNone(agent._state)


if __name__ == "__main__":
    unittest.main()
