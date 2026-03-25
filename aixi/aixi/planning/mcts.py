"""
Family A — minimal root UCT over ξ with explicit revert (IMPLEMENTATION_PLAN §4.1).

Search uses **finite** simulation count and **finite** planning horizon only; returns
are accumulated over a bounded number of imagined (action → percept) cycles.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aixi.aixi.models.mixture import MixtureEnvModel

from aixi.aixi.models.ctw_pyaixi import PyAixiCTWBitMixture


@dataclass(frozen=True)
class MCTSSearchBudget:
    """
    **A1 — documented search budget** (finite truncation; no infinite-horizon semantics).

    - ``mc_simulations``: UCT iterations at the root (each ends by restoring ξ to the
      pre-rollout interaction stream — history replay for ``PyAixiCTWBitMixture``, else paired revert).
    - ``planning_horizon``: imagined (action → percept) cycles per rollout after the
      UCT-selected first action.
    - ``uct_exploration_c``: UCT exploration constant (typical ~sqrt(2)–2).
    - ``discount_gamma``: per-step discount on decoded scalar rewards.
    """

    mc_simulations: int
    planning_horizon: int
    uct_exploration_c: float = 1.4
    discount_gamma: float = 0.99

    def __post_init__(self) -> None:
        if self.mc_simulations < 1:
            raise ValueError("mc_simulations must be >= 1")
        if self.planning_horizon < 1:
            raise ValueError("planning_horizon must be >= 1")
        if self.uct_exploration_c < 0.0:
            raise ValueError("uct_exploration_c must be non-negative")
        if not (0.0 < self.discount_gamma <= 1.0):
            raise ValueError("discount_gamma must be in (0, 1]")


def _sample_next_bit(xi: MixtureEnvModel, rng: Random) -> int:
    p1 = float(xi.predict_bit_probability(1))
    p1 = min(1.0, max(0.0, p1))
    return 1 if rng.random() < p1 else 0


def _sample_percept_and_learn(
    xi: MixtureEnvModel,
    n_percept_bits: int,
    rng: Random,
) -> list[int]:
    symbols: list[int] = []
    for _ in range(n_percept_bits):
        b = _sample_next_bit(xi, rng)
        symbols.append(b)
        xi.learn_symbols([b])
    return symbols


def _rollout_after_first_action(
    xi: MixtureEnvModel,
    *,
    first_action: int,
    encode_action: Callable[[int], Sequence[int]],
    n_action_bits: int,
    n_percept_bits: int,
    decode_reward: Callable[[Sequence[int]], float],
    valid_actions: Sequence[int],
    remaining_steps: int,
    gamma: float,
    rng: Random,
) -> float:
    """
    One imagined (action → percept) step with ``first_action``, then
    ``remaining_steps - 1`` steps with uniform-random actions (playout policy).
    Returns discounted return; mutates ``xi`` — caller must revert.
    """
    g = 0.0
    discount = 1.0

    a_syms = list(encode_action(first_action))
    xi.append_history_symbols(a_syms)
    percept_syms = _sample_percept_and_learn(xi, n_percept_bits, rng)
    r = float(decode_reward(percept_syms))
    g += discount * r
    discount *= gamma

    for _ in range(remaining_steps - 1):
        a = rng.choice(valid_actions)
        a_syms = list(encode_action(a))
        xi.append_history_symbols(a_syms)
        percept_syms = _sample_percept_and_learn(xi, n_percept_bits, rng)
        r = float(decode_reward(percept_syms))
        g += discount * r
        discount *= gamma

    return g


def _revert_imagined_trajectory(
    xi: MixtureEnvModel,
    *,
    n_action_bits: int,
    n_percept_bits: int,
    n_cycles: int,
) -> None:
    """Undo ``n_cycles`` (action → percept) pairs from the model (MC-AIXI undo order)."""
    for _ in range(n_cycles):
        xi.revert_learned_symbols(n_percept_bits)
        xi.revert_history_symbols(n_action_bits)


def root_uct_action(
    xi: MixtureEnvModel,
    budget: MCTSSearchBudget,
    *,
    valid_actions: Sequence[int],
    encode_action: Callable[[int], Sequence[int]],
    n_action_bits: int,
    n_percept_bits: int,
    decode_reward: Callable[[Sequence[int]], float],
    rng: Random,
) -> int:
    """
    Root-only predictive UCT: each simulation picks an arm (legal action), runs a
    bounded-length rollout under ξ (sampled percepts), backs up the return, then
    **fully reverts** ξ to the pre-rollout state.

    Preconditions: same as MC-AIXI ``search()`` — ξ must be ready for an **action**
    (real history should end with a learned percept).
    """
    actions = tuple(valid_actions)
    if not actions:
        raise ValueError("valid_actions must be non-empty")

    for a in actions:
        if len(encode_action(a)) != n_action_bits:
            raise ValueError(
                f"encode_action({a!r}) length must equal n_action_bits={n_action_bits}"
            )

    ref_log_p = xi.root_log_probability()
    # pyaixi: incremental revert after predict-heavy rollouts can desync internal KT weights from
    # ``history`` (extra nodes / wrong root log). Replay from a snapshot instead.
    xi_snap: tuple[int, ...] | None = (
        xi.snapshot_symbol_history() if isinstance(xi, PyAixiCTWBitMixture) else None
    )

    visits: dict[int, int] = defaultdict(int)
    sum_returns: dict[int, float] = defaultdict(float)

    n_sims = budget.mc_simulations
    h = budget.planning_horizon
    c = budget.uct_exploration_c

    for _ in range(n_sims):
        total = sum(visits[a] for a in actions)
        if total == 0:
            chosen = rng.choice(actions)
        else:
            best_a = actions[0]
            best_score = float("-inf")
            log_n = math.log(total + 1e-9)
            for a in actions:
                na = visits[a]
                mean = (sum_returns[a] / na) if na else 0.0
                bonus = c * math.sqrt(log_n / (na + 1e-9)) if na else c * math.sqrt(log_n + 1e-9)
                score = mean + bonus
                if score > best_score:
                    best_score = score
                    best_a = a
            chosen = best_a

        ret = _rollout_after_first_action(
            xi,
            first_action=chosen,
            encode_action=encode_action,
            n_action_bits=n_action_bits,
            n_percept_bits=n_percept_bits,
            decode_reward=decode_reward,
            valid_actions=actions,
            remaining_steps=h,
            gamma=budget.discount_gamma,
            rng=rng,
        )
        if xi_snap is not None:
            assert isinstance(xi, PyAixiCTWBitMixture)
            xi.replay_symbol_history(
                xi_snap,
                n_action_bits=n_action_bits,
                n_percept_bits=n_percept_bits,
            )
        else:
            _revert_imagined_trajectory(
                xi,
                n_action_bits=n_action_bits,
                n_percept_bits=n_percept_bits,
                n_cycles=h,
            )

        visits[chosen] += 1
        sum_returns[chosen] += ret

        if not math.isclose(xi.root_log_probability(), ref_log_p, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(
                "ξ root log P drifted after revert — inner loop must pair learn/revert"
            )

    # Pick most visited (tie-break toward higher mean, then lower action id).
    best = max(
        actions,
        key=lambda a: (visits[a], sum_returns[a] / visits[a] if visits[a] else 0.0, -a),
    )
    return int(best)
