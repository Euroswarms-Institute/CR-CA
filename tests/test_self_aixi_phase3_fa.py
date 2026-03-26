"""Phase 3 (SWA-40): policy + Q function approximation vs Phase 2 tabular baselines."""

from __future__ import annotations

import math
import random

import pytest

try:
    from pyaixi.agents.mc_aixi_ctw import MC_AIXI_CTW_Agent
    from pyaixi.environments.coin_flip import CoinFlip
    from pyaixi.prediction import ctw_context_tree

    from aixi.aixi.models.ctw_pyaixi import PyAixiCTWBitMixture
    from aixi.aixi.parity.coin_flip import coin_flip_options_toy
    from aixi.aixi.planning.self_aixi import (
        SelfAIXIV0Agent,
        SoftDeterministicPolicy,
        UniformRandomPolicy,
    )
    from aixi.aixi.planning.self_aixi.fa import (
        HistoryKeyedQFA,
        fit_softmax_logits_to_policy,
        greedy_action_from_q,
    )

    _HAVE_PYAIXI = True
except ImportError:
    _HAVE_PYAIXI = False

pyaixi = pytest.mark.skipif(not _HAVE_PYAIXI, reason="pyaixi / aixi runtime not available")


@pyaixi
def test_softmax_param_policy_regression_matches_phase2_policies() -> None:
    """Fitted softmax π should match Phase-2 hand policies on the legal action set."""
    rng_fit = random.Random(3)
    valid = (0, 1)
    refs = (
        SoftDeterministicPolicy(preferred=0, slip=0.05),
        SoftDeterministicPolicy(preferred=1, slip=0.05),
        UniformRandomPolicy(),
    )
    for ref in refs:
        pol = fit_softmax_logits_to_policy(ref, valid, rng=rng_fit, steps=5000, lr=0.25)
        for a in valid:
            assert math.isclose(
                pol.prob(a, valid),
                ref.prob(a, valid),
                rel_tol=0.0,
                abs_tol=2e-3,
            )


@pyaixi
def test_history_keyed_q_fa_regression_matches_phase2_greedy() -> None:
    """After each Phase-2 ``act()``, discrete Q FA keyed by ξ history reproduces greedy actions."""
    rng = random.Random(19)
    env = CoinFlip({"coin-flip-p": 0.55})
    mc = MC_AIXI_CTW_Agent(environment=env, options=coin_flip_options_toy())
    xi = PyAixiCTWBitMixture(ctw_context_tree.CTWContextTree(depth=10))
    xi.learn_symbols(mc.encode_percept(env.observation, env.reward))

    policies = (
        SoftDeterministicPolicy(preferred=env.valid_actions[0]),
        SoftDeterministicPolicy(preferred=env.valid_actions[-1]),
        UniformRandomPolicy(),
    )
    agent = SelfAIXIV0Agent(
        xi=xi,
        encode_action=mc.encode_action,
        encode_percept=mc.encode_percept,
        decode_percept=mc.decode_percept,
        valid_actions=tuple(env.valid_actions),
        n_action_bits=env.action_bits(),
        n_percept_bits=env.percept_bits(),
        policies=policies,
        imagination_cycles=3,
        discount_gamma=0.99,
        rng=rng,
    )

    q_fa = HistoryKeyedQFA()
    valid = tuple(env.valid_actions)

    for _ in range(12):
        key = xi.snapshot_symbol_history()
        a_tab = agent.act()
        q_fa.update(key, agent.last_q_by_action)
        q_hat = q_fa.predict(key, valid)
        for a in valid:
            assert math.isclose(q_hat[a], agent.last_q_by_action[a], rel_tol=0.0, abs_tol=1e-9)
        a_hat = greedy_action_from_q(q_hat, valid)
        assert a_hat == a_tab

        agent.append_real_action(a_tab)
        env.perform_action(a_tab)
        agent.learn_real_percept(env.observation, env.reward)
        agent.update_omega(a_tab)


@pyaixi
def test_self_aixi_runs_with_softmax_param_policies() -> None:
    """End-to-end: Phase-3 softmax policies plug into the Phase-2 agent loop."""
    rng = random.Random(31)
    fit_rng = random.Random(5)
    env = CoinFlip({"coin-flip-p": 0.55})
    mc = MC_AIXI_CTW_Agent(environment=env, options=coin_flip_options_toy())
    xi = PyAixiCTWBitMixture(ctw_context_tree.CTWContextTree(depth=10))
    xi.learn_symbols(mc.encode_percept(env.observation, env.reward))

    valid = tuple(env.valid_actions)
    base_policies = (
        SoftDeterministicPolicy(preferred=env.valid_actions[0]),
        SoftDeterministicPolicy(preferred=env.valid_actions[-1]),
        UniformRandomPolicy(),
    )
    policies = tuple(
        fit_softmax_logits_to_policy(p, valid, rng=fit_rng, steps=4000, lr=0.25)
        for p in base_policies
    )
    agent = SelfAIXIV0Agent(
        xi=xi,
        encode_action=mc.encode_action,
        encode_percept=mc.encode_percept,
        decode_percept=mc.decode_percept,
        valid_actions=valid,
        n_action_bits=env.action_bits(),
        n_percept_bits=env.percept_bits(),
        policies=policies,
        imagination_cycles=3,
        discount_gamma=0.99,
        rng=rng,
    )

    for _ in range(8):
        assert math.isclose(sum(agent.omega), 1.0, rel_tol=0.0, abs_tol=1e-9)
        a = agent.act()
        assert a in valid
        agent.append_real_action(a)
        env.perform_action(a)
        agent.learn_real_percept(env.observation, env.reward)
        agent.update_omega(a)
