"""Latent-space planner — random shooting / CEM over short horizons.

Plans entirely in latent space using the AR ``Predictor``, scoring trajectories
with the ``probe``'s HP-difference output. The predictor is autoregressive
over a short history window — at each step we feed the last ``history_size``
embeddings + actions and take the last-position prediction as the next state.
``pred_proj`` is applied to predictor outputs so they live in the same
(post-projector) embedding space the predictor was trained on.

Designed to fit inside 16.67 ms on RTX 3080 with ``num_samples=64``,
``horizon=5``, ``history_size=3``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from leworldgaming.agents.lewm.twohot import twohot_decode


def _repeat_action_blocks(
    actions: torch.Tensor,
    num_actions: int,
    temporal_stride: int,
) -> torch.Tensor:
    one_hot = F.one_hot(actions, num_classes=num_actions).float()
    return (
        one_hot.unsqueeze(-2)
        .expand(*one_hot.shape[:-1], temporal_stride, num_actions)
        .reshape(*one_hot.shape[:-1], temporal_stride * num_actions)
    )


def _concat_action_blocks(
    actions: torch.Tensor,
    num_actions: int,
) -> torch.Tensor:
    """One-hot + concatenate ``temporal_stride`` genuinely distinct
    per-raw-frame actions into a single block embedding input.

    Counterpart to ``_repeat_action_blocks`` (which assumes the *same*
    action was held for the whole block): ``actions`` here has shape
    ``(..., temporal_stride)`` -- one action id per raw frame within the
    block -- matching how Stage-A/B training actually encodes action blocks
    (``_replay_utils.py``'s ``a_oh_raw.reshape(b, steps, stride*action_dim)``
    concatenates the true, possibly-distinct, per-raw-frame recorded
    actions). The planner previously never exploited this -- ``cem_shooting``
    only ever searched over "hold one action for the whole block" sequences,
    an unnecessary restriction of the model's actual expressiveness.
    """
    one_hot = F.one_hot(actions, num_classes=num_actions).float()  # (..., stride, A)
    return one_hot.reshape(*one_hot.shape[:-2], -1)


def _decode_pessimistic(
    head: nn.Module | nn.ModuleList | None,
    bins: torch.Tensor,
    penalty: float,
    *args: torch.Tensor,
) -> torch.Tensor:
    """Decode a (possibly-ensembled) twohot head to a scalar prediction.

    If ``head`` is a plain module, behaves exactly like
    ``twohot_decode(head(*args), bins)`` — the original, non-ensembled path,
    so passing a single ``RewardHead``/``ValueHead`` is fully backward
    compatible.

    If ``head`` is an ``nn.ModuleList`` of independently-initialized heads
    (an ensemble), decodes each member separately and returns
    ``mean - penalty * std`` across members: a pessimistic/lower-confidence
    -bound score. This directly targets "model exploitation" (a CEM planner
    driving trajectories into imagined latents where a single head is
    confidently — and wrongly — optimistic): disagreement across
    independently-trained members is used as a proxy for epistemic
    uncertainty and penalized. ``penalty=0`` degenerates to the plain
    ensemble mean.
    """
    if isinstance(head, nn.ModuleList):
        preds = torch.stack(
            [twohot_decode(member(*args), bins) for member in head], dim=0
        )
        if preds.shape[0] == 1 or penalty == 0.0:
            return preds.mean(dim=0)
        return preds.mean(dim=0) - penalty * preds.std(dim=0)
    return twohot_decode(head(*args), bins)


def _prepare_context(
    z_context: torch.Tensor,
    past_actions: torch.Tensor | None,
    num_samples: int,
    history_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    hs = history_size
    if z_context.ndim == 1:
        z_context = z_context.unsqueeze(0)
    if z_context.shape[0] > hs:
        z_context = z_context[-hs:]
    if z_context.shape[0] < hs:
        pad = z_context[:1].expand(hs - z_context.shape[0], -1)
        z_context = torch.cat([pad, z_context], dim=0)
    z_hist = z_context.unsqueeze(0).expand(num_samples, -1, -1).contiguous()

    if hs == 1:
        past_actions = torch.empty(0, dtype=torch.long, device=device)
    elif past_actions is None:
        past_actions = torch.zeros(hs - 1, dtype=torch.long, device=device)
    else:
        past_actions = past_actions.to(device=device, dtype=torch.long)[-(hs - 1) :]
        if past_actions.numel() < hs - 1:
            pad = torch.zeros(
                hs - 1 - past_actions.numel(), dtype=torch.long, device=device
            )
            past_actions = torch.cat([pad, past_actions])
    action_hist = past_actions.unsqueeze(0).expand(num_samples, -1).contiguous()
    return z_hist, action_hist


@torch.no_grad()
def _score_action_sequences(
    z_hist: torch.Tensor,
    action_hist: torch.Tensor,
    actions: torch.Tensor | None,
    predictor: nn.Module,
    pred_proj: nn.Module,
    action_encoder: nn.Module,
    probe: nn.Module,
    num_actions: int,
    temporal_stride: int,
    reward_head: nn.Module | nn.ModuleList | None,
    continuation_head: nn.Module | None,
    value_head: nn.Module | nn.ModuleList | None,
    reward_bins: torch.Tensor | None,
    value_bins: torch.Tensor | None,
    gamma: float,
    uncertainty_penalty: float = 0.0,
    sub_actions: torch.Tensor | None = None,
    value_weight: float = 1.0,
    reward_clip: float = 0.0,
    idle_action_ids: torch.Tensor | None = None,
    idle_penalty: float = 0.0,
    repeat_penalty: float = 0.0,
    prev_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll ``actions`` (S, H) forward through the latent predictor and return
    a discounted-return score per sample. Shared by ``random_shooting`` (one
    shot) and ``cem_shooting`` (iterative refinement) so both planners score
    trajectories identically.

    ``sub_actions``: optional ``(S, H, temporal_stride)`` tensor of
    genuinely-distinct per-raw-frame action ids for each planned block (see
    ``_concat_action_blocks``). When given, ``actions`` is ignored for
    building the *current* block's embedding (only already-committed history
    still uses the repeated-block encoding, since past decisions really were
    a single action held for the whole block); the block-ending action is
    still folded into ``action_hist`` for conditioning subsequent blocks.

    ``repeat_penalty``: subtract a fixed cost, per planned raw frame, for
    every frame whose action equals the *immediately preceding* raw frame's
    action (within the plan, and against ``prev_action`` -- the actually-
    executed previous decision -- for the very first planned frame). This is
    a deliberate anti-``sticky_prob``/anti-lock-in counterweight: the
    reward/value heads alone often can't distinguish "hold the same stance
    for 100 frames" from "press a fresh useful button" once the score
    landscape is flat (same rationale as ``idle_penalty``, but targeting
    *any* repeated action, not just a fixed no-op set -- e.g. a real live
    failure mode was the agent locking onto one guard/dash action for
    seconds at a time). ``0.0`` (default) disables it, reproducing old
    behavior exactly. ``prev_action``: optional ``(S,)`` tensor of the
    action actually executed on the previous real decision (not merely
    planned) -- lets the very first planned frame be penalized for
    repeating what just really happened, closing the loophole where a
    penalty-free plan could still visibly repeat across decision boundaries.
    """
    if sub_actions is not None:
        s = sub_actions.shape[0]
        horizon = sub_actions.shape[1]
    else:
        s = actions.shape[0]
        horizon = actions.shape[1]
    device = z_hist.device
    scores = torch.zeros(s, device=device)
    discount = torch.ones(s, device=device)
    idle_set = None
    if idle_penalty > 0.0 and idle_action_ids is not None and idle_action_ids.numel() > 0:
        idle_set = idle_action_ids.to(device=device, dtype=torch.long)
    _prev_raw_action = prev_action.to(device=device, dtype=torch.long) if prev_action is not None else None

    for t in range(horizon):
        if sub_actions is not None:
            if action_hist.shape[1] > 0:
                hist_blocks = _repeat_action_blocks(
                    action_hist, num_actions, temporal_stride
                )
            else:
                hist_blocks = torch.zeros(
                    s, 0, temporal_stride * num_actions, device=device
                )
            cur_block = _concat_action_blocks(
                sub_actions[:, t], num_actions
            ).unsqueeze(1)
            action_blocks = torch.cat([hist_blocks, cur_block], dim=1)
        else:
            action_window = torch.cat([action_hist, actions[:, t : t + 1]], dim=1)
            action_blocks = _repeat_action_blocks(
                action_window, num_actions, temporal_stride
            )
        # (#3 idle/no-op penalty) Subtract a fixed cost for every planned
        # frame whose action is a deliberate no-op (e.g. NEUTRAL). The
        # reward/value heads are known-noisy on rare decisive frames (see
        # docs/lewm_calibration_audit_and_ensembling_2026-07-20.md), so a
        # trajectory that just stands still can score approximately as well
        # as one that presses an attack -- CEM has nothing pushing it away
        # from the "safe" degenerate no-op once the score landscape is flat.
        # This is an explicit, deterministic anti-idle bias independent of
        # head calibration. ``idle_penalty <= 0`` (default) disables it,
        # reproducing old behavior exactly.
        if idle_set is not None:
            if sub_actions is not None:
                is_idle = torch.isin(sub_actions[:, t], idle_set).float().mean(dim=-1)
            else:
                is_idle = torch.isin(actions[:, t], idle_set).float()
            scores.sub_(discount * idle_penalty * is_idle)
        if repeat_penalty > 0.0:
            if sub_actions is not None:
                cur_raw = sub_actions[:, t]  # (S, temporal_stride)
                if t == 0:
                    prev_raw = _prev_raw_action.unsqueeze(-1) if _prev_raw_action is not None else None
                else:
                    prev_raw = sub_actions[:, t - 1, -1:]
                if prev_raw is not None:
                    lead = torch.cat([prev_raw, cur_raw[:, :-1]], dim=-1)
                    is_repeat = (cur_raw == lead).float().mean(dim=-1)
                    scores.sub_(discount * repeat_penalty * is_repeat)
            else:
                if t == 0:
                    prev_a = _prev_raw_action
                else:
                    prev_a = actions[:, t - 1]
                if prev_a is not None:
                    is_repeat = (actions[:, t] == prev_a).float()
                    scores.sub_(discount * repeat_penalty * is_repeat)
        a_hist_emb = action_encoder(action_blocks)
        if reward_head is not None and reward_bins is not None:
            step_reward = _decode_pessimistic(
                reward_head, reward_bins, uncertainty_penalty,
                z_hist[:, -1], a_hist_emb[:, -1],
            )
            # (#2) Winsorize per-step reward: cap the magnitude of any single
            # imagined frame's reward so CEM cannot chase one hallucinated
            # big hit into an unreliable latent. ``reward_clip <= 0`` disables.
            if reward_clip > 0.0:
                step_reward = step_reward.clamp(-reward_clip, reward_clip)
            scores.add_(discount * step_reward)
        z_pred_seq = predictor(z_hist, a_hist_emb)  # (S, HS, D)
        z_next = pred_proj(z_pred_seq[:, -1])  # (S, D)
        if continuation_head is not None:
            discount.mul_(
                gamma * torch.sigmoid(continuation_head(z_next))
            )
        else:
            discount.mul_(gamma)
        z_hist = torch.cat([z_hist[:, 1:], z_next.unsqueeze(1)], dim=1)
        if sub_actions is not None:
            action_hist = torch.cat(
                [action_hist[:, 1:], sub_actions[:, t, -1:]], dim=1
            )
        else:
            action_hist = action_window[:, 1:]

    final_z = z_hist[:, -1]  # (S, D)
    if value_head is not None and value_bins is not None:
        # (#2) Down-weight the terminal value head -- it is the noisiest part
        # of the score on this game, so trusting it at weight 1.0 lets CEM
        # exploit its miscalibration. ``value_weight`` (default 1.0) is a
        # planner-only knob; 0.3-0.7 leans on per-step reward instead.
        scores.add_(
            value_weight
            * discount
            * _decode_pessimistic(value_head, value_bins, uncertainty_penalty, final_z)
        )
    elif reward_head is None:
        scores = scores + probe(final_z)[:, 0]
    return scores


@torch.no_grad()
def random_shooting(
    z_context: torch.Tensor,
    predictor: nn.Module,
    pred_proj: nn.Module,
    action_encoder: nn.Module,
    probe: nn.Module,
    num_actions: int,
    horizon: int = 5,
    num_samples: int = 64,
    history_size: int = 3,
    temporal_stride: int = 1,
    past_actions: torch.Tensor | None = None,
    reward_head: nn.Module | nn.ModuleList | None = None,
    continuation_head: nn.Module | None = None,
    value_head: nn.Module | nn.ModuleList | None = None,
    reward_bins: torch.Tensor | None = None,
    value_bins: torch.Tensor | None = None,
    gamma: float = 0.997,
    valid_actions: torch.Tensor | None = None,
    uncertainty_penalty: float = 0.0,
) -> int:
    """Sample ``num_samples`` i.i.d. random action sequences of length
    ``horizon``, roll out in latent space, score, return the first action of
    the best sequence.

    Kept as the legacy/baseline planner. Every call resamples from scratch
    with no memory of the previous decision's plan, which — combined with a
    noisy/miscalibrated scoring head — tends to produce visibly erratic,
    non-committal behavior (the policy "flails" instead of executing a
    coherent short combo). Prefer ``cem_shooting`` for live play.

    ``valid_actions``: optional 1D long tensor restricting sampling to a
    subset of ``[0, num_actions)`` (see ``LewmAgent``'s
    ``restrict_to_playable_actions`` — roughly a third of FightingICE's raw
    ``Action`` enum are unplayable "state observation" values like
    ``STAND``/``AIR``/``*_RECOV``/``THROW_HIT`` that ``CommandCenter`` never
    maps to a key combo and that the replay data never recorded as an
    executed action, so their action-embedding is untrained noise).
    """
    device = z_context.device
    z_hist, action_hist = _prepare_context(
        z_context, past_actions, num_samples, history_size, device
    )
    if valid_actions is not None:
        idx = torch.randint(0, valid_actions.numel(), (num_samples, horizon), device=device)
        actions = valid_actions.to(device)[idx]
    else:
        actions = torch.randint(0, num_actions, (num_samples, horizon), device=device)
    scores = _score_action_sequences(
        z_hist, action_hist, actions, predictor, pred_proj, action_encoder, probe,
        num_actions, temporal_stride, reward_head, continuation_head, value_head,
        reward_bins, value_bins, gamma, uncertainty_penalty,
    )
    best = int(scores.argmax().item())
    return int(actions[best, 0].item())


@torch.no_grad()
def cem_shooting(
    z_context: torch.Tensor,
    predictor: nn.Module,
    pred_proj: nn.Module,
    action_encoder: nn.Module,
    probe: nn.Module,
    num_actions: int,
    horizon: int = 5,
    num_samples: int = 64,
    num_iters: int = 3,
    elite_frac: float = 0.125,
    sticky_prob: float = 0.5,
    momentum: float = 0.3,
    min_prob: float = 0.02,
    history_size: int = 3,
    temporal_stride: int = 1,
    past_actions: torch.Tensor | None = None,
    reward_head: nn.Module | nn.ModuleList | None = None,
    continuation_head: nn.Module | None = None,
    value_head: nn.Module | nn.ModuleList | None = None,
    reward_bins: torch.Tensor | None = None,
    value_bins: torch.Tensor | None = None,
    gamma: float = 0.997,
    init_dist: torch.Tensor | None = None,
    valid_actions: torch.Tensor | None = None,
    uncertainty_penalty: float = 0.0,
    warm_shift: int = 1,
    plan_raw_actions: bool = False,
    elite_temp: float = 0.0,
    value_weight: float = 1.0,
    reward_clip: float = 0.0,
    idle_action_ids: torch.Tensor | None = None,
    idle_penalty: float = 0.0,
    policy_prior: torch.Tensor | None = None,
    repeat_penalty: float = 0.0,
    prev_action: int | None = None,
) -> tuple[int, torch.Tensor]:
    """Discrete iCEM-style planner (Pinneri et al., "Sample-efficient
    Cross-Entropy Method for Real-time Planning", CoRL 2020, arXiv:2008.06389).

    Two changes vs. plain ``random_shooting`` that directly target
    "the policy just runs around erratically" symptoms without touching the
    trained world model:

    1. **Iterative elite refinement** (proper CEM instead of one-shot random
       shooting): maintain a per-timestep categorical action distribution,
       sample, keep the top ``elite_frac`` trajectories by score, refit the
       distribution towards them, repeat for ``num_iters``. This exploits the
       (frozen) predictor/heads far better than scoring 64 uniform-random
       sequences once.
    2. **Warm-starting + "sticky" (colored-noise-style) sampling**: the
       distribution from the *previous* decision is shifted and reused as
       this decision's prior (instead of resetting to uniform every frame),
       and within a rollout each timestep repeats the previous timestep's
       sampled action with probability ``sticky_prob`` instead of drawing an
       independent fresh action. This is the discrete analogue of the
       temporally-correlated ("colored") noise iCEM uses for continuous
       control — it stops trajectories (and hence the executed policy) from
       jittering between unrelated actions every single frame.

    Also directly sidesteps a miscalibrated head: pass
    ``continuation_head=None`` to force a constant ``gamma`` discount instead
    of trusting a head known to be badly calibrated on held-out data (see
    ``docs/opponent_conditioning_research_2026-07-16.md``).

    3. **Sample, don't argmax, the executed action**: the earlier version
       returned ``dist[0].argmax()``, which made the fixed point of the
       warm-start/momentum loop deterministic — once the mode locked onto
       one action, sampling+elites+refit kept reinforcing it with nothing to
       escape it (observed: 200+ consecutive live decisions of the same
       action, i.e. several real seconds where the character just holds one
       pose/stance and looks frozen or "stuck defending"). Categorical
       sampling from the final ``dist[0]`` keeps the exploitation benefit
       (the mode is still by far the most likely draw) while leaving a
       genuine per-decision chance to escape a bad lock-in — the discrete
       analogue of MPPI's soft Boltzmann-weighted update vs. a hard
       top-k/argmax choice.

    Returns ``(action, updated_distribution)`` — the caller should persist
    ``updated_distribution`` and pass it back in as ``init_dist`` on the next
    call (via ``LewmAgent._plan_dist``) to keep the warm start across frames.

    ``valid_actions``: optional 1D long tensor of action indices to restrict
    both the uniform prior/floor and the multinomial sampling to (see
    ``random_shooting``'s docstring for why — untrained "state observation"
    action indices should never be sampled or reinforced).

    ``uncertainty_penalty``: if ``reward_head``/``value_head`` are passed as
    ``nn.ModuleList`` ensembles (trained via ``train_lewm_heads.py``'s
    ``reward_ensemble_size``/``value_ensemble_size``), each scored trajectory
    uses ``mean - uncertainty_penalty * std`` across ensemble members instead
    of a single head's point estimate — a pessimistic lower-confidence-bound
    that discourages the planner from exploiting imagined trajectories only
    one member is confidently (and possibly wrongly) optimistic about. Has no
    effect when passed a plain (non-ensembled) module. ``0.0`` (default) is
    the plain ensemble mean with no pessimism.

    ``warm_shift``: how many horizon steps to drop off the front of
    ``init_dist`` before reusing it as this call's prior. The default ``1``
    is correct when this planner runs every decision (one step executed
    per call, as originally designed). Callers that only invoke this
    planner every ``N`` decisions and otherwise replay a cached
    multi-step action chunk sampled from the previous ``dist`` (see
    ``LewmAgent.act``'s ``chunk_size`` option, used to cut inference cost
    and real-time frame drops) should pass ``warm_shift=N`` so the warm
    start accounts for all ``N`` already-executed steps instead of just one.
    When ``plan_raw_actions=True``, ``warm_shift`` (and ``dist``'s row
    index) is in raw-frame units, not block units -- see below.

    ``plan_raw_actions``: if ``True``, search over ``temporal_stride``
    genuinely distinct actions per planned block instead of assuming one
    action is held for the whole block (the original/default behavior,
    unchanged when this is ``False``). Every block *was* actually recorded
    with real, possibly-distinct, per-raw-frame actions during Stage-A/B
    training (see ``_concat_action_blocks``); forcing the planner to only
    ever consider "hold one action for `temporal_stride` frames" sequences
    was an unnecessary restriction that can't express short combos (e.g.
    crouch then punch) within a single block. Internally, ``dist`` becomes
    ``(horizon * temporal_stride, num_actions)`` -- one row per *raw* frame
    across the whole planned span, in raw-frame execution order -- instead
    of ``(horizon, num_actions)``. ``dist[0]`` is still the very next raw
    action to execute, exactly as in the default mode, so callers (e.g.
    ``LewmAgent.act`` with ``chunk_size>1``) can keep dequeuing
    ``dist[1], dist[2], ...`` unchanged to get the rest of the *current*
    block's distinct raw actions, one real action per raw frame -- this
    only makes sense if the caller also requests a fresh decision from the
    environment every raw frame (``frame_skip=1``) rather than holding one
    action for ``temporal_stride`` frames itself, since holding would
    flatten the distinction right back out.
    """
    device = z_context.device
    s = num_samples
    elite_k = max(1, int(round(s * elite_frac)))

    z_hist, action_hist = _prepare_context(
        z_context, past_actions, s, history_size, device
    )

    _prev_action_tensor = None
    if repeat_penalty > 0.0 and prev_action is not None:
        _prev_action_tensor = torch.full((s,), int(prev_action), dtype=torch.long, device=device)

    # ``plan_len`` is the number of rows ``dist`` maintains: one per planned
    # *block* (default) or one per planned *raw frame* across the whole span
    # (``plan_raw_actions=True``, ``plan_len = horizon * temporal_stride``).
    plan_len = horizon * temporal_stride if plan_raw_actions else horizon

    uniform_row = torch.zeros(num_actions, device=device)
    if valid_actions is not None:
        valid_actions = valid_actions.to(device=device, dtype=torch.long)
        uniform_row[valid_actions] = 1.0 / valid_actions.numel()
    else:
        uniform_row.fill_(1.0 / num_actions)

    # (#4 policy-prior warm start, TD-MPC2/Sampled-MuZero style) Replace the
    # blind-uniform initial row with a behavior-cloned prior over actions
    # (see policy_head.py) when supplied. This ONLY changes what CEM's
    # first iteration samples from (and the exploration floor every
    # subsequent iteration, in place of ``uniform_row``) -- every sampled
    # trajectory is still scored by the exact same reward/value heads and
    # dist is still refit from real scores every iteration, so a
    # miscalibrated/stale prior can only cost a couple of extra
    # elite-refinement iterations to correct, never silently override
    # planner judgment. Falls back to ``uniform_row`` exactly (identical
    # behavior to before this feature existed) when ``policy_prior is None``.
    init_row = uniform_row
    if policy_prior is not None:
        init_row = policy_prior.to(device=device, dtype=uniform_row.dtype)
        if valid_actions is not None:
            # Re-mask + renormalize over only the valid/playable actions so
            # an untrained/leaky prior can never put mass on unplayable
            # "state observation" ids (see agent.py's _commandable_action_ids
            # docstring for why that's a known failure mode).
            mask = torch.zeros_like(init_row)
            mask[valid_actions] = 1.0
            init_row = init_row * mask
        total = init_row.sum()
        if total > 0:
            init_row = init_row / total
        else:
            init_row = uniform_row

    if init_dist is None:
        dist = init_row.unsqueeze(0).expand(plan_len, -1).contiguous()
    else:
        # Warm start: shift the previous plan forward by the number of
        # steps already executed since it was computed (1 in the original
        # per-decision-replan design; >1 when running in chunked mode) and
        # pad fresh prior-row(s) at the end for the newly-exposed horizon.
        shift = max(1, int(warm_shift))
        dist = torch.cat(
            [init_dist.to(device)[shift:], init_row.unsqueeze(0).expand(min(shift, plan_len), -1)],
            dim=0,
        )

    for _ in range(max(1, num_iters)):
        actions = torch.zeros((s, plan_len), dtype=torch.long, device=device)
        for t in range(plan_len):
            fresh = torch.multinomial(
                dist[t].unsqueeze(0).expand(s, -1), 1
            ).squeeze(-1)
            if t == 0 or sticky_prob <= 0.0:
                actions[:, t] = fresh
            else:
                stick = torch.rand(s, device=device) < sticky_prob
                actions[:, t] = torch.where(stick, actions[:, t - 1], fresh)

        if plan_raw_actions:
            sub_actions = actions.view(s, horizon, temporal_stride)
            scores = _score_action_sequences(
                z_hist, action_hist, None, predictor, pred_proj, action_encoder,
                probe, num_actions, temporal_stride, reward_head, continuation_head,
                value_head, reward_bins, value_bins, gamma, uncertainty_penalty,
                sub_actions=sub_actions,
                value_weight=value_weight, reward_clip=reward_clip,
                idle_action_ids=idle_action_ids, idle_penalty=idle_penalty,
                repeat_penalty=repeat_penalty, prev_action=_prev_action_tensor,
            )
        else:
            scores = _score_action_sequences(
                z_hist, action_hist, actions, predictor, pred_proj, action_encoder,
                probe, num_actions, temporal_stride, reward_head, continuation_head,
                value_head, reward_bins, value_bins, gamma, uncertainty_penalty,
                value_weight=value_weight, reward_clip=reward_clip,
                idle_action_ids=idle_action_ids, idle_penalty=idle_penalty,
                repeat_penalty=repeat_penalty, prev_action=_prev_action_tensor,
            )
        elite_idx = scores.topk(min(elite_k, s)).indices
        elite_actions = actions[elite_idx]  # (elite_k, plan_len)

        new_dist = torch.zeros((plan_len, num_actions), device=device)
        if elite_temp > 0.0:
            # (#1) Soft / Boltzmann elite update (MPPI-style): instead of a
            # hard 0/1 top-k count, weight the elites by softmax(score/temp)
            # so the full score ranking within the elite set shapes the
            # refit. Far more sample-efficient at low ``num_samples`` than a
            # hard cutoff -- the standard reason MPPI beats vanilla CEM.
            elite_scores = scores[elite_idx]  # (elite_k,)
            w = torch.softmax(elite_scores / elite_temp, dim=0)  # (elite_k,)
            for t in range(plan_len):
                oh = F.one_hot(elite_actions[:, t], num_classes=num_actions).float()
                new_dist[t] = (w.unsqueeze(-1) * oh).sum(dim=0)
        else:
            for t in range(plan_len):
                counts = torch.bincount(elite_actions[:, t], minlength=num_actions).float()
                new_dist[t] = counts / counts.sum().clamp_min(1.0)
        new_dist = (1.0 - min_prob) * new_dist + min_prob * uniform_row
        dist = momentum * dist + (1.0 - momentum) * new_dist
        dist = dist / dist.sum(dim=-1, keepdim=True)

    # Sample (not argmax) the executed action from the final dist[0].
    #
    # Taking argmax here turned this into a deterministic fixed point: the
    # warm-started `dist` is blended (`momentum`) frame-to-frame, and the
    # "new" component each call is itself refit from elites drawn from that
    # same warm-started distribution. Once dist[0]'s mode locks onto one
    # action there is nothing left to break the cycle -- sampling, elite
    # selection, and refitting all keep reinforcing the same mode, and argmax
    # removes the one thing (stochastic escape) that could exit it. Observed
    # in practice as the executed policy repeating one action for hundreds of
    # consecutive decisions (looks "stuck"/idle in a live match). Categorical
    # sampling keeps CEM's exploitation benefit (the mode is still by far the
    # most likely draw) while leaving a real, non-zero chance of escaping a
    # bad lock-in every single decision -- the discrete analogue of MPPI's
    # soft (Boltzmann-weighted) update instead of a hard top-k/argmax choice.
    action0 = int(torch.multinomial(dist[0], 1).item())
    return action0, dist.detach()
