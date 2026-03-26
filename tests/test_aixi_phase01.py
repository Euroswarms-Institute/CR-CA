"""Phase 0–1: pyaixi parity and MixtureEnvModel + multi-symbol revert stress."""

from __future__ import annotations

import math
import random

import pytest

try:
    from pyaixi.prediction import ctw_context_tree

    from aixi.aixi.models.ctw_pyaixi import PyAixiCTWBitMixture
    from aixi.aixi.parity.coin_flip import run_coin_flip_symbol_parity

    _HAVE_PYAIXI = True
except ImportError:
    _HAVE_PYAIXI = False

pyaixi = pytest.mark.skipif(not _HAVE_PYAIXI, reason="pyaixi / aixi runtime not available")


@pyaixi
def test_coin_flip_symbol_parity_agent_vs_manual_ctw() -> None:
    rows = run_coin_flip_symbol_parity(seed=7, cycles=25)
    for agent_lp, manual_lp in rows:
        assert math.isclose(agent_lp, manual_lp, rel_tol=0.0, abs_tol=1e-9)


@pyaixi
def test_py_aixi_ctw_adapter_implements_mixture_protocol() -> None:
    tree = ctw_context_tree.CTWContextTree(depth=6)
    xi = PyAixiCTWBitMixture(tree)
    assert xi.model_ids == ("pyaixi-ctw",)
    xi.append_history_symbols([0, 1, 0])
    xi.learn_symbols([1])
    p1 = xi.predict_bit_probability(1)
    assert 0.0 <= p1 <= 1.0
    _ = xi.root_log_probability()


@pyaixi
def test_m3_revert_stress_learned_path_restores_root_log_p() -> None:
    """M3: triple (3-symbol) revert blocks; stack several to stress CTW rollback."""
    rng = random.Random(11)
    tree = ctw_context_tree.CTWContextTree(depth=12)
    xi = PyAixiCTWBitMixture(tree)

    for _ in range(8):
        snapshot = xi.root_log_probability()
        hist_len = len(tree.history)
        triple = [rng.randint(0, 1) for _ in range(3)]
        xi.learn_symbols(triple)
        xi.revert_learned_symbols(3)
        assert math.isclose(xi.root_log_probability(), snapshot, rel_tol=0.0, abs_tol=1e-9)
        assert len(tree.history) == hist_len


@pyaixi
def test_m3_mixed_revert_learned_then_history_matches_agent_undo_pattern() -> None:
    """Match MC-AIXI undo: percept uses ``revert(percept_bits)``, action uses ``revert_history``."""
    from pyaixi.environments.coin_flip import CoinFlip

    env = CoinFlip({"coin-flip-p": 0.5})
    percept_bits = env.reward_bits() + env.observation_bits()
    action_bits = env.action_bits()

    tree = ctw_context_tree.CTWContextTree(depth=16)
    xi = PyAixiCTWBitMixture(tree)

    rng = random.Random(3)
    for _ in range(12):
        ref_log = xi.root_log_probability()
        ref_hist = list(tree.history)

        action_syms = [rng.randint(0, 1) for _ in range(action_bits)]
        xi.append_history_symbols(action_syms)
        percept_syms = [rng.randint(0, 1) for _ in range(percept_bits)]
        xi.learn_symbols(percept_syms)

        xi.revert_learned_symbols(percept_bits)
        xi.revert_history_symbols(action_bits)

        assert math.isclose(xi.root_log_probability(), ref_log, rel_tol=0.0, abs_tol=1e-9)
        assert tree.history == ref_hist


@pyaixi
def test_phase1_unified_xi_rollout_uniform_matches_explicit_uniform_policy() -> None:
    """MCTS (implicit uniform) and Self-AIXI ``UniformRandomPolicy`` share ``xi_rollouts``."""
    from pyaixi.agents.mc_aixi_ctw import MC_AIXI_CTW_Agent
    from pyaixi.environments.coin_flip import CoinFlip

    from aixi.aixi.parity.coin_flip import coin_flip_options_toy
    from aixi.aixi.planning.self_aixi import UniformRandomPolicy
    from aixi.aixi.planning.xi_rollouts import (
        imagined_trajectory_discounted_return,
        restore_mixture_after_imagination,
    )

    env = CoinFlip({"coin-flip-p": 0.62})
    mc = MC_AIXI_CTW_Agent(environment=env, options=coin_flip_options_toy())
    depth = 12
    tree_a = ctw_context_tree.CTWContextTree(depth=depth)
    xi_a = PyAixiCTWBitMixture(tree_a)
    xi_a.learn_symbols(mc.encode_percept(env.observation, env.reward))
    snap = xi_a.snapshot_symbol_history()

    tree_b = ctw_context_tree.CTWContextTree(depth=depth)
    xi_b = PyAixiCTWBitMixture(tree_b)
    xi_b.replay_symbol_history(
        snap,
        n_action_bits=env.action_bits(),
        n_percept_bits=env.percept_bits(),
    )
    assert math.isclose(
        xi_a.root_log_probability(),
        xi_b.root_log_probability(),
        rel_tol=0.0,
        abs_tol=1e-9,
    )

    uni = UniformRandomPolicy()
    actions = tuple(env.valid_actions)
    ab, pb = env.action_bits(), env.percept_bits()
    n_cycles = 4
    gamma = 0.99

    def decode_reward(syms):
        _o, r = mc.decode_percept(list(syms))
        return float(r)

    seed = 12345
    for first_a in actions:
        rng_a = random.Random(seed)
        rng_b = random.Random(seed)
        ref_a = xi_a.root_log_probability()
        ref_b = xi_b.root_log_probability()
        snap_a = xi_a.snapshot_symbol_history()
        snap_b = xi_b.snapshot_symbol_history()

        g_a = imagined_trajectory_discounted_return(
            xi_a,
            first_action=first_a,
            encode_action=mc.encode_action,
            n_action_bits=ab,
            n_percept_bits=pb,
            decode_reward=decode_reward,
            valid_actions=actions,
            n_cycles=n_cycles,
            gamma=gamma,
            rng=rng_a,
            subsequent_action=None,
        )
        restore_mixture_after_imagination(
            xi_a,
            xi_snap=snap_a,
            ref_log_p=ref_a,
            n_action_bits=ab,
            n_percept_bits=pb,
            n_cycles=n_cycles,
        )

        g_b = imagined_trajectory_discounted_return(
            xi_b,
            first_action=first_a,
            encode_action=mc.encode_action,
            n_action_bits=ab,
            n_percept_bits=pb,
            decode_reward=decode_reward,
            valid_actions=actions,
            n_cycles=n_cycles,
            gamma=gamma,
            rng=rng_b,
            subsequent_action=lambda r, acts, u=uni: u.sample_action(r, acts),
        )
        restore_mixture_after_imagination(
            xi_b,
            xi_snap=snap_b,
            ref_log_p=ref_b,
            n_action_bits=ab,
            n_percept_bits=pb,
            n_cycles=n_cycles,
        )

        assert math.isclose(g_a, g_b, rel_tol=0.0, abs_tol=1e-9)
        assert math.isclose(
            xi_a.root_log_probability(),
            xi_b.root_log_probability(),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
