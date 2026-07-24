"""Online opponent model — RHEAPI-style live-adapting opponent predictor.

Motivation (see docs/lewm_planner_literature_research_2026-07-22.md):
RHEAPI (Tang et al., IEEE T-Games 2020), the strongest published non-hardcoded
agent on this exact platform, won its ~10-30 win-rate-point margin *not* from
its search algorithm but from an **online-learned opponent model** trained live
at round boundaries. Our LeWM planner has no opponent model at all — it rolls
the latent predictor forward conditioned only on *our* action, implicitly
assuming an average/marginal opponent, and cannot anticipate the specific
reactive behaviour of the live Dreamer opponent.

A *faithful* RHEAPI port would feed the predicted opponent action into the
forward model every rollout step. Our latent ``Predictor`` has no
opponent-action input channel (it was trained single-agent, ``p(z' | z,
a_own)``), so that faithful path needs a data-recollection + retrain (blocked
tonight by ~0.6 fps headless pixel-capture throughput — see the research doc).

This module implements the *no-retrain* variant that the same literature
supports: an online-trained **threat model** ``p(the opponent lands damage on
us within the next decision block | current relative/opponent state)``, learned
by online logistic SGD from the *directly observed* outcome each decision (did
our HP drop?). Unlike the reward/value heads (which the calibration audit found
imprecise exactly on the rare decisive frames), this signal is:

  * **fresh & well-calibrated** — trained on-the-fly on the actual opponent
    being faced, from ground-truth HP deltas, not offline replay of other
    opponents;
  * **cheap** — a ~10-feature logistic unit, updated once per decision, no
    pixels, no GPU rollout;
  * **adaptive across rounds** — exactly RHEAPI's "no benefit round 1,
    improves in later rounds" behaviour, since it accumulates the opponent's
    observed damage pattern.

The threat probability is turned into a *first-action bias* over the planner's
final action distribution (see ``LewmAgent`` / ``planner`` integration): when a
hit is predicted imminent, guard/evasion actions are up-weighted and committal
attacks down-weighted; when the opponent is predicted passive/recovering, our
attacks are up-weighted. The counter-*mapping* is domain knowledge, but its
*trigger* is the online-learned, well-calibrated threat estimate — the part
that the current miscalibrated heads cannot provide.

This is deliberately a small, ablatable overlay (fully off by default): it does
not touch the trained world model, the heads, or the CEM search itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Playable action-id groupings (FightingICE Action enum ints). Kept module-level
# and lazily validated so this file imports even without pyftg present.
_GUARD_IDS: tuple[int, ...] = (10, 11, 12)          # STAND/CROUCH/AIR_GUARD
_EVASION_IDS: tuple[int, ...] = (2, 6, 4)           # BACK_STEP, BACK_JUMP, JUMP
# Attack ids (STAND/CROUCH/AIR normals, specials, throws) — the committal
# actions we want to suppress when a hit is imminent and boost when it's safe.
_ATTACK_IDS: tuple[int, ...] = (
    23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class OnlineOpponentModel:
    """Online logistic threat predictor + a threat-conditioned action bias.

    ``strength`` scales how hard the threat estimate biases the planner's
    first-action distribution. ``lr`` is the online SGD step size. ``threshold``
    is the neutral threat level (0.5) around which the bias is signed.

    All state is per-episode-resettable via ``reset()`` but the *learned
    weights persist across rounds* (RHEAPI's key property: it keeps improving
    against the same opponent as the match goes on). Call ``reset()`` only on a
    brand-new opponent/match if you want a cold model.
    """

    strength: float = 1.5
    lr: float = 0.02
    threshold: float = 0.5
    l2: float = 1e-4
    # Feature dimension: see ``_raw_features`` (6 features + 1 bias weight = 7).
    _dim: int = 7
    w: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    # Rolling stats for logging/debug.
    n_updates: int = 0
    _last_feat: np.ndarray | None = None
    _last_hp_own: float | None = None
    _last_p: float = 0.5
    # Running feature normalization (online mean/var) to keep the logistic
    # input well-scaled without a calibration pass.
    _feat_mean: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    _feat_var: np.ndarray = field(default_factory=lambda: np.ones(6, dtype=np.float64))
    _feat_count: float = 1.0

    def reset(self) -> None:
        """Reset per-episode transient state (NOT the learned weights)."""
        self._last_feat = None
        self._last_hp_own = None
        self._last_p = 0.5

    # ---- feature extraction ------------------------------------------------
    @staticmethod
    def _raw_features(obs: dict[str, Any]) -> np.ndarray:
        """Threat-relevant features from a ``frame_to_obs_dict`` observation.

        Uses only the opponent + relative geometry — the quantities that
        determine whether the opponent can damage us in the next ~block.
        Robust to missing keys (returns zeros) so it never crashes ``act``.
        """
        own = obs.get("own", {}) or {}
        opp = obs.get("opp", {}) or {}
        g = obs.get("global", {}) or {}
        max_hp = float(g.get("max_hp", 400) or 400)

        def _f(d: dict[str, Any], k: str, default: float = 0.0) -> float:
            v = d.get(k, default)
            try:
                return float(v)
            except Exception:
                return default

        own_x = _f(own, "x")
        opp_x = _f(opp, "x")
        dist = abs(opp_x - own_x)
        opp_attacking = 1.0 if int(_f(opp, "action")) in _ATTACK_IDS else 0.0
        opp_atk_live = _f(opp, "atk_is_live")
        # Small startup = attack about to become active = maximally dangerous.
        opp_atk_startup = _f(opp, "atk_start_up")
        startup_imminent = 1.0 if (opp_atk_live > 0.5 and opp_atk_startup <= 8) else 0.0
        opp_control = _f(opp, "control")
        proj_opp = _f(g, "proj_opp")

        # Deliberately excludes slow-drifting quantities (hp_diff, energy) and
        # weak ones (vertical offset): on a monotone-HP stretch a drifting
        # feature becomes a spurious dominant predictor and hijacks the online
        # fit (observed in a synthetic check). These 6 are all instantaneous
        # threat geometry -- they reset to their true value every frame.
        return np.array(
            [
                dist / 400.0,          # normalized horizontal gap
                opp_attacking,         # opp's current action is an attack
                opp_atk_live,          # opp has a live attack object
                startup_imminent,      # live attack with <=8f startup
                opp_control,           # opp is actionable (can start a move)
                proj_opp,              # opp projectile in flight
            ],
            dtype=np.float64,
        )

    def _normalize(self, raw: np.ndarray, update_stats: bool) -> np.ndarray:
        # Features are already ~0-1 scaled by construction; an online
        # mean/var normalizer was tried and removed -- constant features
        # (std->0) blew up their normalized value and destabilized the
        # logistic fit (informative-feature weights collapsed to ~0 in a
        # synthetic check). Identity keeps the fit well-conditioned.
        return raw

    # ---- online prediction + update ---------------------------------------
    def predict_threat(self, obs: dict[str, Any]) -> float:
        """Return p(opponent damages us in the next block) for this obs.

        Also stashes the feature vector + current HP so the *next* call to
        ``observe_outcome`` can form a supervised (feature, did-we-get-hit)
        pair and take one online SGD step.
        """
        raw = self._raw_features(obs)
        feat = self._normalize(raw, update_stats=True)
        z = float(np.dot(self.w[:-1], feat) + self.w[-1])
        p = _sigmoid(z)
        self._last_feat = feat
        self._last_p = p
        own = obs.get("own", {}) or {}
        try:
            self._last_hp_own = float(own.get("hp"))
        except Exception:
            self._last_hp_own = None
        return p

    def observe_outcome(self, obs: dict[str, Any]) -> None:
        """Close the loop: use the HP change since the last ``predict_threat``
        as the binary label (1 = we took damage) and take one logistic SGD
        step. Call once per decision, *before* the next ``predict_threat``.
        """
        if self._last_feat is None or self._last_hp_own is None:
            return
        own = obs.get("own", {}) or {}
        try:
            hp_now = float(own.get("hp"))
        except Exception:
            return
        took_damage = 1.0 if hp_now < self._last_hp_own - 1e-6 else 0.0
        feat = self._last_feat
        z = float(np.dot(self.w[:-1], feat) + self.w[-1])
        p = _sigmoid(z)
        grad_scale = p - took_damage  # dL/dz for logistic BCE
        # SGD with L2 on the feature weights (not the bias).
        self.w[:-1] -= self.lr * (grad_scale * feat + self.l2 * self.w[:-1])
        self.w[-1] -= self.lr * grad_scale
        self.n_updates += 1

    # ---- action-distribution bias -----------------------------------------
    def bias_action_dist(
        self,
        dist_row: np.ndarray,
        threat_p: float,
        valid_action_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reweight a first-action probability row by the threat estimate.

        ``dist_row``: 1D array over the full action space (sums to 1).
        Returns a renormalized copy. When ``threat_p`` is high, guard/evasion
        actions are multiplicatively boosted and attacks suppressed; when low,
        attacks are boosted. Magnitude scales with |threat_p - threshold| and
        ``strength``. Actions outside ``valid_action_ids`` are untouched (they
        should already be ~0 from the planner's restriction).
        """
        signed = threat_p - self.threshold  # >0 => threatened, <0 => safe
        mag = self.strength * abs(signed)
        if mag < 1e-6:
            return dist_row
        out = dist_row.astype(np.float64).copy()
        if signed > 0:
            # Hit imminent: favor defense/evasion, suppress committal attacks.
            defend_factor = math.exp(mag)
            attack_factor = math.exp(-mag)
            for i in _GUARD_IDS + _EVASION_IDS:
                if i < out.shape[0]:
                    out[i] *= defend_factor
            for i in _ATTACK_IDS:
                if i < out.shape[0]:
                    out[i] *= attack_factor
        else:
            # Safe / opponent recovering: favor our offense.
            attack_factor = math.exp(mag)
            for i in _ATTACK_IDS:
                if i < out.shape[0]:
                    out[i] *= attack_factor
        s = out.sum()
        if s <= 0:
            return dist_row
        return out / s

    def debug_state(self) -> dict[str, float]:
        return {
            "n_updates": float(self.n_updates),
            "last_threat_p": float(self._last_p),
            "bias_weight": float(self.w[-1]),
            "w_norm": float(np.linalg.norm(self.w[:-1])),
        }


def bias_action_dist_from_opp_prediction(
    dist_row: np.ndarray,
    opp_action_probs: np.ndarray,
    strength: float = 1.5,
) -> np.ndarray:
    """Reweight our first-action distribution using a *real, BC-trained*
    prediction of the opponent's next action (``opp_action_probs``, from
    ``agents/lewm/opp_action_head.OppActionHead`` -- trained on genuine
    recorded Dreamer opponent actions, see
    ``scripts/train_opp_action_head.py`` / ``docs/lewm_opp_action_head_2026-07-23.md``).

    This is the *real-data* counterpart to ``OnlineOpponentModel``'s online
    logistic threat model (which used hand-picked geometric proxy features,
    not a real opponent-action label, since no opponent-conditioned data
    existed until this session's fresh collection). Same counter-mapping
    domain knowledge (attack predicted -> favor guard/evasion; else -> favor
    our offense) but the *trigger* is now p(opponent attacks) summed over
    ``_ATTACK_IDS`` from the trained classifier's softmax, instead of an
    online-fit threat probability from instantaneous threat-geometry features.

    ``dist_row``: 1D array over the full action space (sums to 1).
    ``opp_action_probs``: 1D softmax array over the same action space (the
    OppActionHead's prediction for the opponent's next action).
    Returns a renormalized copy of ``dist_row``.
    """
    n = dist_row.shape[0]
    p_opp_attack = float(
        sum(opp_action_probs[i] for i in _ATTACK_IDS if i < opp_action_probs.shape[0])
    )
    signed = p_opp_attack - 0.5  # >0 => opponent probably attacking
    mag = strength * abs(signed)
    if mag < 1e-6:
        return dist_row
    out = dist_row.astype(np.float64).copy()
    if signed > 0:
        defend_factor = math.exp(mag)
        attack_factor = math.exp(-mag)
        for i in _GUARD_IDS + _EVASION_IDS:
            if i < n:
                out[i] *= defend_factor
        for i in _ATTACK_IDS:
            if i < n:
                out[i] *= attack_factor
    else:
        attack_factor = math.exp(mag)
        for i in _ATTACK_IDS:
            if i < n:
                out[i] *= attack_factor
    s = out.sum()
    if s <= 0:
        return dist_row
    return out / s
