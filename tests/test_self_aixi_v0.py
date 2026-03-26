"""Self-AIXI v0: ω over finite policies + ξ imagination (IMPLEMENTATION_PLAN §4.2)."""

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
    from aixi.aixi.planning.self_aixi import SelfAIXIV0Agent, SoftDeterministicPolicy, UniformRandomPolicy
    from aixi.aixi.planning.self_aixi.metrics import snapshot_mc_vs_self_identical_xi

    _HAVE_PYAIXI = True
except ImportError:
    _HAVE_PYAIXI = False

pyaixi = pytest.mark.skipif(not _HAVE_PYAIXI, reason="pyaixi / aixi runtime not available")


@pyaixi
def test_self_aixi_v0_loop_invariants_coin_flip() -> None:
    rng = random.Random(7)
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

    for _ in range(6):
        assert math.isclose(sum(agent.omega), 1.0, rel_tol=0.0, abs_tol=1e-9)
        a = agent.act()
        assert a in env.valid_actions
        assert set(agent.last_q_by_action.keys()) == set(env.valid_actions)
        assert len(agent.last_q_pi) == len(env.valid_actions)
        assert all(len(row) == len(policies) for row in agent.last_q_pi)

        agent.append_real_action(a)
        env.perform_action(a)
        agent.learn_real_percept(env.observation, env.reward)
        agent.update_omega(a)

    assert math.isclose(sum(agent.omega), 1.0, rel_tol=0.0, abs_tol=1e-9)


@pyaixi
def test_self_aixi_suboptimality_log_vs_mc_aixi_shared_xi() -> None:
    """Phase 2 (SWA-39): identical ξ; log decision mismatch and Q_{ωξ} gap vs MC-AIXI action."""
    from pyaixi.agent import action_update

    rng = random.Random(11)
    env = CoinFlip({"coin-flip-p": 0.55})
    opts = {**coin_flip_options_toy(), "mc-simulations": 24}
    mc = MC_AIXI_CTW_Agent(environment=env, options=opts)
    mc.reset()
    assert mc.last_update == action_update
    mc.model_update_percept(env.observation, env.reward)

    xi = PyAixiCTWBitMixture(mc.context_tree)
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
        imagination_cycles=4,
        discount_gamma=0.99,
        rng=rng,
    )

    subopt_log: list[tuple[bool, float]] = []
    for _ in range(10):
        snap = snapshot_mc_vs_self_identical_xi(mc=mc, self_agent=agent)
        subopt_log.append((snap.actions_match, snap.q_gap_argmax_vs_mc_choice))
        assert snap.q_gap_argmax_vs_mc_choice >= -1e-9

        a_exec = snap.action_self_aixi
        mc.model_update_action(a_exec)
        env.perform_action(a_exec)
        mc.model_update_percept(env.observation, env.reward)
        agent.append_real_action(a_exec)
        agent.learn_real_percept(env.observation, env.reward)
        agent.update_omega(a_exec)

    assert len(subopt_log) == 10
    assert all(isinstance(t, bool) and isinstance(g, float) for t, g in subopt_log)
