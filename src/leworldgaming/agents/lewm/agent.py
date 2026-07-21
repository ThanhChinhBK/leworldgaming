"""LeWM agent — encoder + projector + AR predictor + pred_proj + action encoder + probe + planner.

After Stage-B head training (``train_lewm_heads.py``) the checkpoint also
carries reward / continuation / value heads. They are optional at load
time: if the ckpt has only Stage-A keys the heads stay at random init and
``act()`` falls back to the legacy random-shooting planner.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn

from leworldgaming.agents.base import AgentBase
from leworldgaming.agents.lewm.action_encoder import ActionEncoder
from leworldgaming.agents.lewm.continuation_head import ContinuationHead
from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.mcts_planner import mcts_search
from leworldgaming.agents.lewm.planner import cem_shooting, random_shooting
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.probe import LinearProbe
from leworldgaming.agents.lewm.projector import Projector
from leworldgaming.agents.lewm.reward_head import RewardHead
from leworldgaming.agents.lewm.twohot import make_bins
from leworldgaming.agents.lewm.value_head import ValueHead


def _commandable_action_ids(num_actions: int) -> torch.Tensor | None:
    """Indices of ``[0, num_actions)`` that ``CommandCenter.action_to_command``
    actually maps to a key combo, plus ``NEUTRAL`` (a deliberate no-op).

    FightingICE's raw ``pyftg.Action`` enum (56 values matching the default
    ``action_dim``) also contains ~16 "state observation" values —
    ``STAND``/``AIR``/``*_GUARD_RECOV``/``*_RECOV``/``CHANGE_DOWN``/``DOWN``/
    ``RISE``/``LANDING``/``THROW_HIT``/``THROW_SUFFER`` — that describe what
    the character is currently doing, not a command a player can issue.
    ``RecordingAI`` only ever records the actually-requested (playable)
    action as training-data labels (see ``env/recording_ai.py``), so these
    ~16 indices' rows in ``ActionEncoder`` are essentially never trained —
    their embeddings are close to random init.

    If the (frozen, Stage-A-trained) planner ever samples one of these as
    "best", ``CommandCenter.command_call`` silently falls through every
    ``elif`` branch, ``skill_key`` stays empty, and the frame gets a blank
    ``Key()`` — i.e. this decision window does *nothing*, no matter what the
    game state was. Under one-shot ``random_shooting`` this happens on
    ~28% of samples; even under CEM, once one of these low-signal actions
    gets an inflated score from noise it can dominate a whole elite set.
    This is a likely major contributor to LeWM looking like it "wanders"
    instead of committing to a coherent, aggressive plan.

    Returns ``None`` (no restriction — falls back to uniform-over-everything,
    matching old behavior) if ``num_actions`` doesn't match the real 56-way
    FightingICE action space, since the mapping below is only valid for it.
    """
    if num_actions != 56:
        return None
    try:
        from pyftg.models.enums.action import Action

        from leworldgaming.env.policies import PLAYABLE_ACTIONS
    except Exception:  # pragma: no cover - pyftg optional in some test envs
        return None
    playable_names = {a.name for a in PLAYABLE_ACTIONS}
    ids = [
        i for i in range(num_actions)
        if Action.from_int(i).name == "NEUTRAL" or Action.from_int(i).name in playable_names
    ]
    return torch.tensor(ids, dtype=torch.long)


class LewmAgent(AgentBase):
    """Inference-time wrapper around the trained modules.

    Architecture is configured by a flat ``cfg`` dict matching the keys in
    ``configs/lewm.yaml`` / ``train_lewm.DEFAULTS``. ``load()`` rebuilds
    modules from the checkpoint's stored config so the agent's architecture
    always matches the saved weights — no need to remember training flags.
    """

    def __init__(self, cfg: dict[str, Any] | None = None, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self._build_modules(cfg or {})

    def _build_modules(self, cfg: dict[str, Any]) -> None:
        self.model_cfg = dict(cfg)
        self.action_dim = int(cfg.get("action_dim", 56))
        self.temporal_stride = int(cfg.get("temporal_stride", 1) or 1)
        latent_dim = int(cfg.get("latent_dim", 192))
        self.latent_dim = latent_dim
        self.history_size = int(cfg.get("history_size", 3))
        self.image_size = int(cfg.get("encoder_image_size", 224))
        projector_hidden = int(cfg.get("projector_hidden", 2048))
        self.heads_cfg: dict[str, Any] = dict(cfg.get("heads", {}))
        self.heads_loaded = False  # set True only after Stage-B weights load.

        self.encoder = Encoder(
            latent_dim=latent_dim,
            image_size=int(cfg.get("encoder_image_size", 224)),
            patch_size=int(cfg.get("encoder_patch_size", 14)),
            embed_dim=int(cfg.get("encoder_embed_dim", 192)),
            depth=int(cfg.get("encoder_depth", 12)),
            num_heads=int(cfg.get("encoder_heads", 3)),
            mlp_ratio=float(cfg.get("encoder_mlp_ratio", 4.0)),
            dropout=float(cfg.get("encoder_dropout", 0.0)),
        ).to(self.device)
        self.projector = Projector(latent_dim=latent_dim, hidden_dim=projector_hidden).to(
            self.device
        )
        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim * self.temporal_stride, emb_dim=latent_dim
        ).to(self.device)
        self.predictor = Predictor(
            latent_dim=latent_dim,
            action_dim=latent_dim,
            history_size=self.history_size,
            depth=int(cfg.get("predictor_depth", 6)),
            num_heads=int(cfg.get("predictor_heads", 16)),
            dim_head=int(cfg.get("predictor_dim_head", 64)),
            mlp_dim=int(cfg.get("predictor_mlp_dim", 2048)),
            dropout=float(cfg.get("predictor_dropout", 0.1)),
        ).to(self.device)
        self.pred_proj = Projector(latent_dim=latent_dim, hidden_dim=projector_hidden).to(
            self.device
        )
        probe_targets = self.heads_cfg.get("probe_targets", [47, 0, 22, 46])
        self.probe = LinearProbe(latent_dim=latent_dim, target_dim=len(probe_targets)).to(
            self.device
        )

        head_hidden = int(self.heads_cfg.get("hidden_dim", 512))
        reward_bins = int(self.heads_cfg.get("reward_bins", 41))
        value_bins = int(self.heads_cfg.get("value_bins", 41))
        # Ensemble sizes (2026-07-20): >1 builds an nn.ModuleList of
        # independently-initialized heads instead of a single module.
        # planner.py's ``_decode_pessimistic`` detects nn.ModuleList and
        # scores with mean-minus-uncertainty across members instead of a
        # single point estimate. 1 (default) is the original single-head
        # behavior, unchanged.
        n_reward = max(1, int(self.heads_cfg.get("reward_ensemble_size", 1) or 1))
        n_value = max(1, int(self.heads_cfg.get("value_ensemble_size", 1) or 1))
        self.reward_head: nn.Module | nn.ModuleList
        if n_reward > 1:
            self.reward_head = nn.ModuleList(
                [
                    RewardHead(latent_dim=latent_dim, hidden_dim=head_hidden, num_bins=reward_bins)
                    for _ in range(n_reward)
                ]
            ).to(self.device)
        else:
            self.reward_head = RewardHead(
                latent_dim=latent_dim, hidden_dim=head_hidden, num_bins=reward_bins
            ).to(self.device)
        # Continuation head may use a narrower hidden width than the shared
        # reward/value heads (heads_cfg["cont_hidden_dim"]) -- see
        # train_lewm_heads.py / ContinuationHead docstring: with only ~300
        # terminal-window training examples, a smaller head overfits less.
        # Falls back to the shared head_hidden for older checkpoints that
        # don't set this key.
        cont_hidden = int(self.heads_cfg.get("cont_hidden_dim") or head_hidden)
        self.continuation_head = ContinuationHead(
            latent_dim=latent_dim, hidden_dim=cont_hidden
        ).to(self.device)
        self.value_head: nn.Module | nn.ModuleList
        if n_value > 1:
            self.value_head = nn.ModuleList(
                [
                    ValueHead(latent_dim=latent_dim, hidden_dim=head_hidden, num_bins=value_bins)
                    for _ in range(n_value)
                ]
            ).to(self.device)
        else:
            self.value_head = ValueHead(
                latent_dim=latent_dim, hidden_dim=head_hidden, num_bins=value_bins
            ).to(self.device)
        self.reward_bins = make_bins(
            reward_bins,
            float(self.heads_cfg.get("reward_low", -1.0)),
            float(self.heads_cfg.get("reward_high", 1.0)),
            self.device,
        )
        self.value_bins = make_bins(
            value_bins,
            float(self.heads_cfg.get("value_low", -10.0)),
            float(self.heads_cfg.get("value_high", 10.0)),
            self.device,
        )
        # Policy-guided CEM was tried (2026-07-21) and abandoned: the
        # distilled Q-value target across all ~56 actions turned out to be
        # essentially uniform (std ~0.0008, entropy ~= log(num_valid) even
        # at temperature=0.01), i.e. the reward/value heads don't yet
        # discriminate actions well enough on a single step to distill a
        # useful prior. No policy-prior head/config remains.
        self._z_history: list[torch.Tensor] = []
        self._action_history: list[int] = []

        # Planner knobs — all overridable via cfg["planner"]. Defaults pick
        # the iCEM-style planner (see planner.cem_shooting) with the
        # continuation head DISABLED at inference: the stride-5 continuation
        # head is known to be badly miscalibrated on held-out terminal
        # windows (see docs/opponent_conditioning_research_2026-07-16.md and
        # repo memory "lewm continuation head"), and letting a noisy signal
        # multiplicatively discount every rollout is a major source of the
        # planner picking visibly erratic/non-committal actions. None of
        # this requires retraining Stage A or Stage B — it's inference-time
        # search-quality + robustness only.
        #
        # Defaults tuned for latency: the real-time budget per decision is
        # `temporal_stride * 16.67ms` (60fps). At temporal_stride=5 that's
        # ~83ms/decision. The naive samples=64/iters=3/horizon=5 CEM config
        # costs ~70ms/decision on an RTX 5060 Ti (measured), leaving almost
        # no slack for pixel capture/encoding or GPU contention with an
        # opponent agent -- decisions arrive late and the agent looks
        # sluggish/reactive-only vs a ~3ms Dreamer actor forward pass.
        # samples=24/iters=1/horizon=8 costs ~38ms (measured, ~46% of the
        # stride=5 budget) -- prioritizes a much longer lookahead (~0.67s
        # vs ~0.17s at horizon=2) over extra samples/iters, since a single
        # scoring pass with warm-started/sticky elite selection (via the
        # carried-over `_plan_dist`) already captures most of CEM's benefit
        # over plain random_shooting at this sample count.
        #
        # At temporal_stride=2 the budget shrinks to ~33ms/decision --
        # horizon=8/samples=24 (~38-41ms measured, INCLUDING real capture
        # overhead) blows this budget and causes every decision to be
        # dropped under --pace realtime (observed: 100% frame-budget drops
        # in a live stride=2 test, meaning the agent always played one
        # stale action behind -- see docs/lewm_stride2_retrain_decision_
        # 2026-07-17.md). horizon=5/samples=20 costs ~25-28ms (measured,
        # ~15-20% margin under budget) -- shorter lookahead (~167ms vs the
        # ~330ms originally hoped for) but reliably decides on time, which
        # matters far more than search depth given a late decision is
        # simply discarded.
        _stride_planner_defaults = {2: (5, 20)}.get(self.temporal_stride, (8, 24))
        # sticky_prob/momentum/min_prob also need a stride=2-specific tuning:
        # the same warm-start/momentum/sticky-repeat design that worked fine
        # at stride=5 (samples=24, elite_k=3) becomes a near-guaranteed
        # action lock-in at stride=2's tighter budget (samples=20,
        # elite_k=2) -- less exploration mass per decision to escape a
        # locked mode. Combined with the argmax->sample fix in
        # ``planner.cem_shooting`` (see its docstring point 3), stride=2
        # gets a faster-decaying warm start (lower momentum), milder
        # temporal correlation (lower sticky_prob, closer to iCEM's actual
        # colored-noise strength rather than a 50% hard repeat), and a
        # higher exploration floor (higher min_prob) so a bad lock-in both
        # forms less easily and is escaped faster once it does.
        _stride_cem_tuning = {2: (0.2, 0.1, 0.05)}.get(
            self.temporal_stride, (0.5, 0.3, 0.02)
        )
        planner_cfg: dict[str, Any] = dict(cfg.get("planner", {}))
        self.planner_name = str(planner_cfg.get("name", "cem"))
        self.planner_horizon = int(planner_cfg.get("horizon", _stride_planner_defaults[0]))
        self.planner_num_samples = int(
            planner_cfg.get("num_samples", _stride_planner_defaults[1])
        )
        self.planner_num_iters = int(planner_cfg.get("num_iters", 1))
        self.planner_elite_frac = float(planner_cfg.get("elite_frac", 0.125))
        self.planner_sticky_prob = float(
            planner_cfg.get("sticky_prob", _stride_cem_tuning[0])
        )
        self.planner_momentum = float(planner_cfg.get("momentum", _stride_cem_tuning[1]))
        self.planner_min_prob = float(planner_cfg.get("min_prob", _stride_cem_tuning[2]))
        # MCTS/PUCT-only knobs (see mcts_planner.mcts_search). Unused when
        # planner_name != "mcts".
        self.planner_num_simulations = int(planner_cfg.get("num_simulations", 24))
        self.planner_max_depth = int(planner_cfg.get("max_depth", self.planner_horizon))
        self.planner_c_puct = float(planner_cfg.get("c_puct", 1.25))
        self.planner_dirichlet_alpha = float(planner_cfg.get("dirichlet_alpha", 0.3))
        self.planner_dirichlet_frac = float(planner_cfg.get("dirichlet_frac", 0.25))
        self.planner_temperature = float(planner_cfg.get("temperature", 1.0))
        # Wave/virtual-loss batching knobs (see mcts_planner's module
        # docstring) -- control how many simulations' leaves are evaluated
        # per batched NN call. Larger sim_batch_size = fewer, bigger calls
        # (faster) at the cost of slightly less-informed within-wave
        # selection.
        self.planner_sim_batch_size = int(planner_cfg.get("sim_batch_size", 16))
        self.planner_virtual_loss = float(planner_cfg.get("virtual_loss", 1.0))
        self.use_continuation_head = bool(planner_cfg.get("use_continuation_head", False))
        self.use_value_head = bool(planner_cfg.get("use_value_head", True))
        # Pessimistic ensemble scoring (see planner._decode_pessimistic):
        # only has any effect when reward_head/value_head are ModuleList
        # ensembles (reward_ensemble_size/value_ensemble_size > 1). 0.0
        # (plain ensemble mean, no pessimism) preserves old behavior when
        # ensembles aren't in use.
        self.planner_uncertainty_penalty = float(
            planner_cfg.get("uncertainty_penalty", 0.0)
        )
        # Chunked (open-loop) execution: instead of re-running the full
        # CEM search every decision, only replan once every ``chunk_size``
        # decisions, executing the rest of that decision's already-computed
        # multi-step ``dist`` (see ``planner.cem_shooting``) without any
        # extra encoder/predictor/planner compute in between. Motivated by
        # the measured stride=2 CEM cost (~34ms) sitting right at the ~33ms
        # per-decision real-time budget (causing ~49% dropped decisions,
        # see docs/lewm_calibration_audit_and_ensembling_2026-07-20.md) --
        # replanning every ``chunk_size`` decisions instead multiplies the
        # effective per-decision budget by ``chunk_size`` while keeping the
        # exact same planner compute cost per replan. Trades some
        # reactivity (mid-chunk decisions replay a stale plan instead of
        # reacting to the latest frame) for eliminating frame drops
        # entirely; must be validated live, not assumed. ``1`` (default)
        # is the original per-decision-replan behavior, unchanged.
        self.planner_chunk_size = max(1, int(planner_cfg.get("chunk_size", 1)))
        # Distinct per-raw-frame action planning within a block (see
        # planner.cem_shooting's plan_raw_actions doc): searches over
        # temporal_stride genuinely different actions per planned block
        # instead of assuming one action is held for the whole block. Only
        # useful combined with chunk_size==temporal_stride and the caller
        # requesting a decision from the environment every raw frame
        # (frame_skip=1) -- otherwise env-side frame_skip repetition would
        # flatten the distinct actions back into one held action anyway.
        self.planner_plan_raw_actions = bool(
            planner_cfg.get("plan_raw_actions", False)
        )
        self.restrict_to_playable_actions = bool(
            planner_cfg.get("restrict_to_playable_actions", True)
        )
        self._valid_action_ids = (
            _commandable_action_ids(self.action_dim)
            if self.restrict_to_playable_actions
            else None
        )
        self._plan_dist: torch.Tensor | None = None
        # Cached actions from the last replan, sampled from dist[1:], not
        # yet consumed by act(); and how many decisions have executed
        # since the last actual cem_shooting call (fed to its warm_shift).
        self._chunk_queue: list[int] = []
        self._chunk_consumed: int = 0

        self._set_eval()

    def _set_eval(self) -> None:
        for m in (
            self.encoder,
            self.projector,
            self.action_encoder,
            self.predictor,
            self.pred_proj,
            self.probe,
            self.reward_head,
            self.continuation_head,
            self.value_head,
        ):
            m.eval()

    @torch.no_grad()
    def act(self, obs: dict[str, Any]) -> int:
        x = obs["pixels"].to(self.device)
        z = self.projector(self.encoder(x.unsqueeze(0))).squeeze(0)
        z_context = torch.stack([*self._z_history, z], dim=0)
        past_actions = torch.as_tensor(
            self._action_history, dtype=torch.long, device=self.device
        )
        use_reward = self.heads_loaded and float(
            self.heads_cfg.get("reward_loss_weight", 0.0)
        ) > 0.0
        use_cont = (
            self.heads_loaded
            and self.use_continuation_head
            and float(self.heads_cfg.get("cont_loss_weight", 0.0)) > 0.0
        )
        use_value = (
            self.heads_loaded
            and self.use_value_head
            and float(self.heads_cfg.get("value_loss_weight", 0.0)) > 0.0
        )
        common_kwargs = dict(
            predictor=self.predictor,
            pred_proj=self.pred_proj,
            action_encoder=self.action_encoder,
            probe=self.probe,
            num_actions=self.action_dim,
            history_size=self.history_size,
            temporal_stride=self.temporal_stride,
            past_actions=past_actions,
            reward_head=self.reward_head if use_reward else None,
            continuation_head=self.continuation_head if use_cont else None,
            value_head=self.value_head if use_value else None,
            reward_bins=self.reward_bins,
            value_bins=self.value_bins,
            gamma=float(self.heads_cfg.get("gamma", 0.997)),
            valid_actions=self._valid_action_ids,
            uncertainty_penalty=self.planner_uncertainty_penalty,
        )
        # ``horizon``/``num_samples`` are cem_shooting/random_shooting-only
        # kwargs (mcts_search uses num_simulations/max_depth instead) --
        # kept out of the shared dict above and added per-branch below.
        if self.planner_name == "cem":
            if self._chunk_queue:
                # Mid-chunk: reuse an action already sampled from the last
                # replan's multi-step ``dist`` instead of re-running the
                # (expensive) search. Skips predictor/head compute entirely
                # for this decision -- only the encoder call above (needed
                # to keep _z_history/_action_history fresh for the next
                # actual replan) runs.
                action = self._chunk_queue.pop(0)
                self._chunk_consumed += 1
            else:
                action, self._plan_dist = cem_shooting(
                    z_context,
                    horizon=self.planner_horizon,
                    num_samples=self.planner_num_samples,
                    num_iters=self.planner_num_iters,
                    elite_frac=self.planner_elite_frac,
                    sticky_prob=self.planner_sticky_prob,
                    momentum=self.planner_momentum,
                    min_prob=self.planner_min_prob,
                    init_dist=self._plan_dist,
                    warm_shift=max(self._chunk_consumed, 1),
                    plan_raw_actions=self.planner_plan_raw_actions,
                    **common_kwargs,
                )
                self._chunk_consumed = 0
                if self.planner_chunk_size > 1:
                    extra = min(self.planner_chunk_size - 1, self._plan_dist.shape[0] - 1)
                    self._chunk_queue = [
                        int(torch.multinomial(self._plan_dist[k], 1).item())
                        for k in range(1, extra + 1)
                    ]
        elif self.planner_name == "mcts":
            # Discrete MCTS/PUCT (see mcts_planner.mcts_search) -- a
            # genuinely different search algorithm from CEM (explicit tree,
            # PUCT-guided simulation allocation) rather than a variant of
            # it. Every real decision restarts a fresh tree from the
            # actually-observed state (no warm-start/chunking analogue yet).
            action, _ = mcts_search(
                z_context,
                num_simulations=self.planner_num_simulations,
                max_depth=self.planner_max_depth,
                c_puct=self.planner_c_puct,
                dirichlet_alpha=self.planner_dirichlet_alpha,
                dirichlet_frac=self.planner_dirichlet_frac,
                temperature=self.planner_temperature,
                sim_batch_size=self.planner_sim_batch_size,
                virtual_loss=self.planner_virtual_loss,
                **common_kwargs,
            )
        else:
            action = random_shooting(
                z_context,
                horizon=self.planner_horizon,
                num_samples=self.planner_num_samples,
                **common_kwargs,
            )
        self._z_history.append(z.detach())
        self._action_history.append(action)
        keep = max(self.history_size - 1, 0)
        if keep:
            self._z_history = self._z_history[-keep:]
            self._action_history = self._action_history[-keep:]
        else:
            self._z_history.clear()
            self._action_history.clear()
        return action

    def reset_episode(self) -> None:
        self._z_history.clear()
        self._action_history.clear()
        self._plan_dist = None
        self._chunk_queue = []
        self._chunk_consumed = 0

    def configure_planner(
        self,
        name: str | None = None,
        horizon: int | None = None,
        num_samples: int | None = None,
        num_iters: int | None = None,
        elite_frac: float | None = None,
        sticky_prob: float | None = None,
        momentum: float | None = None,
        min_prob: float | None = None,
        use_continuation_head: bool | None = None,
        use_value_head: bool | None = None,
        restrict_to_playable_actions: bool | None = None,
        uncertainty_penalty: float | None = None,
        chunk_size: int | None = None,
        plan_raw_actions: bool | None = None,
        num_simulations: int | None = None,
        max_depth: int | None = None,
        c_puct: float | None = None,
        dirichlet_alpha: float | None = None,
        dirichlet_frac: float | None = None,
        temperature: float | None = None,
        sim_batch_size: int | None = None,
        virtual_loss: float | None = None,
    ) -> None:
        """Override planner choice/hyperparameters after construction or ``load()``.

        ``load()`` rebuilds modules (and resets planner settings to their
        cfg defaults) from the checkpoint's own saved config, so this must be
        called *after* ``load()`` to stick. Lets callers (``scripts/play.py``,
        ``scripts/self_play.py``) switch between ``"random"`` (the original
        JEPA-planning-paper baseline), ``"cem"`` (iCEM-style, see
        ``planner.cem_shooting``), and ``"mcts"`` (discrete MCTS/PUCT, see
        ``mcts_planner.mcts_search``) per run without retraining anything.
        """
        if name is not None:
            if name not in ("random", "cem", "mcts"):
                raise ValueError(f"Unknown planner: {name!r}. Choose: random, cem, mcts.")
            self.planner_name = name
        if horizon is not None:
            self.planner_horizon = int(horizon)
        if num_samples is not None:
            self.planner_num_samples = int(num_samples)
        if num_iters is not None:
            self.planner_num_iters = int(num_iters)
        if elite_frac is not None:
            self.planner_elite_frac = float(elite_frac)
        if sticky_prob is not None:
            self.planner_sticky_prob = float(sticky_prob)
        if momentum is not None:
            self.planner_momentum = float(momentum)
        if min_prob is not None:
            self.planner_min_prob = float(min_prob)
        if use_continuation_head is not None:
            self.use_continuation_head = bool(use_continuation_head)
        if use_value_head is not None:
            self.use_value_head = bool(use_value_head)
        if uncertainty_penalty is not None:
            self.planner_uncertainty_penalty = float(uncertainty_penalty)
        if chunk_size is not None:
            self.planner_chunk_size = max(1, int(chunk_size))
        if plan_raw_actions is not None:
            self.planner_plan_raw_actions = bool(plan_raw_actions)
        if num_simulations is not None:
            self.planner_num_simulations = int(num_simulations)
        if max_depth is not None:
            self.planner_max_depth = int(max_depth)
        if c_puct is not None:
            self.planner_c_puct = float(c_puct)
        if dirichlet_alpha is not None:
            self.planner_dirichlet_alpha = float(dirichlet_alpha)
        if dirichlet_frac is not None:
            self.planner_dirichlet_frac = float(dirichlet_frac)
        if temperature is not None:
            self.planner_temperature = float(temperature)
        if sim_batch_size is not None:
            self.planner_sim_batch_size = max(1, int(sim_batch_size))
        if virtual_loss is not None:
            self.planner_virtual_loss = float(virtual_loss)
        if restrict_to_playable_actions is not None:
            self.restrict_to_playable_actions = bool(restrict_to_playable_actions)
            self._valid_action_ids = (
                _commandable_action_ids(self.action_dim)
                if self.restrict_to_playable_actions
                else None
            )
        self._plan_dist = None
        self._chunk_queue = []
        self._chunk_consumed = 0

    @torch.no_grad()
    def warmup(self, n_iters: int = 2) -> None:
        """Run dummy ``act()`` calls to JIT-compile MPS/CUDA kernels.

        First-time PyTorch forward passes on a fresh shape can take 1–3 s
        while the backend compiles kernels — long enough to stall the JVM
        and look like a frozen game. Call ``warmup()`` once after ``load()``
        and before ``gateway.run_game()`` to pay that cost up front.
        """
        dummy = torch.zeros(
            (3, self.image_size, self.image_size),
            device=self.device,
            dtype=torch.float32,
        )
        for _ in range(int(n_iters)):
            self.act({"pixels": dummy})

    def learn(self, batch: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError(
            "Use leworldgaming.training.train_lewm.train() for the offline JEPA loop."
        )

    def save(self, path: str) -> None:
        save_dict = {
            "encoder": self.encoder.state_dict(),
            "projector": self.projector.state_dict(),
            "action_encoder": self.action_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "pred_proj": self.pred_proj.state_dict(),
            "probe": self.probe.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "continuation_head": self.continuation_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "heads_config": self.heads_cfg,
            "config": self.model_cfg,
        }
        torch.save(save_dict, path)

    @staticmethod
    def _load_head(head: nn.Module | nn.ModuleList, ckpt: dict[str, Any], plural_key: str, singular_key: str) -> None:
        """Load a (possibly-ensembled) head's weights from either checkpoint format.

        ``train_lewm_heads.py`` checkpoints (ensemble-aware, 2026-07-20+)
        save both: ``plural_key`` (a list of per-member state dicts) and
        ``singular_key`` (member 0 only, for older non-ensemble-aware
        consumers). ``LewmAgent.save()`` and pre-ensembling checkpoints only
        have ``singular_key``. Prefers the plural key when both ``head`` is
        an ensemble and the checkpoint has it; otherwise falls back to the
        singular key loaded into ``head`` directly (single module) or its
        first member (ensemble of size 1).
        """
        if isinstance(head, nn.ModuleList):
            if plural_key in ckpt:
                saved = ckpt[plural_key]
                if len(saved) != len(head):
                    raise ValueError(
                        f"Checkpoint has {len(saved)} {plural_key} members but this "
                        f"agent was configured with {len(head)} — reward_ensemble_size/"
                        "value_ensemble_size must match the checkpoint's heads_config."
                    )
                for member, state in zip(head, saved, strict=True):
                    member.load_state_dict(state)
            elif len(head) == 1 and singular_key in ckpt:
                head[0].load_state_dict(ckpt[singular_key])
            else:
                raise KeyError(
                    f"Checkpoint has neither {plural_key!r} nor a compatible {singular_key!r}."
                )
        else:
            head.load_state_dict(ckpt[singular_key])

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config")
        # Stage-B checkpoints carry heads_config; fold it into cfg so
        # _build_modules sizes the heads correctly before loading weights.
        if cfg is not None and "heads_config" in ckpt:
            cfg = dict(cfg)
            cfg["heads"] = ckpt["heads_config"]
        if cfg:
            self._build_modules(cfg)
        self.encoder.load_state_dict(ckpt["encoder"])
        if "projector" in ckpt:
            self.projector.load_state_dict(ckpt["projector"])
        if "action_encoder" in ckpt:
            self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        if "pred_proj" in ckpt:
            self.pred_proj.load_state_dict(ckpt["pred_proj"])

        stage_b_keys = ("reward_head", "continuation_head", "value_head")
        has_stage_b = all(k in ckpt for k in stage_b_keys)
        if "probe" in ckpt:
            self.probe.load_state_dict(ckpt["probe"])
        if has_stage_b:
            self._load_head(self.reward_head, ckpt, "reward_heads", "reward_head")
            self.continuation_head.load_state_dict(ckpt["continuation_head"])
            self._load_head(self.value_head, ckpt, "value_heads", "value_head")
            self.heads_loaded = True
        else:
            warnings.warn(
                "LewmAgent.load: Stage-B heads (reward/continuation/value) not found in "
                f"{path}; reward/value heads are at random init. MCTS planning will not "
                "work — train Stage B via train_lewm_heads.py.",
                stacklevel=2,
            )
        self._set_eval()
