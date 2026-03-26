"""Phase 2: root UCT + ξ rollouts with revert (Family A, IMPLEMENTATION_PLAN §4.1)."""

from __future__ import annotations

import math
import random

import pytest

from aixi.aixi.planning.mcts import MCTSSearchBudget, root_uct_action

try:
    from pyaixi.agents.mc_aixi_ctw import MC_AIXI_CTW_Agent
    from pyaixi.environments.coin_flip import CoinFlip
    from pyaixi.prediction import ctw_context_tree

    from aixi.aixi.env.pyaixi_adapter import PyaixiEnvAdapter
    from aixi.aixi.env.two_armed_bandit_pyaixi import TwoArmedBandit
    from aixi.aixi.models.ctw_pyaixi import PyAixiCTWBitMixture
    from aixi.aixi.parity.coin_flip import coin_flip_options_toy

    _HAVE_PYAIXI = True
except ImportError:
    _HAVE_PYAIXI = False

pyaixi = pytest.mark.skipif(not _HAVE_PYAIXI, reason="pyaixi / aixi runtime not available")


def test_mcts_search_budget_a1_validation() -> None:
    with pytest.raises(ValueError, match="mc_simulations"):
        MCTSSearchBudget(mc_simulations=0, planning_horizon=2)
    with pytest.raises(ValueError, match="planning_horizon"):
        MCTSSearchBudget(mc_simulations=1, planning_horizon=0)


@pyaixi
def test_root_uct_leaves_xi_root_log_p_unchanged_coin_flip() -> None:
    rng = random.Random(42)
    env = CoinFlip({"coin-flip-p": 0.55})
    agent = MC_AIXI_CTW_Agent(environment=env, options=coin_flip_options_toy())
    tree = ctw_context_tree.CTWContextTree(depth=10)
    xi = PyAixiCTWBitMixture(tree)

    xi.learn_symbols(agent.encode_percept(env.observation, env.reward))
    ref = xi.root_log_probability()

    budget = MCTSSearchBudget(mc_simulations=12, planning_horizon=3, uct_exploration_c=1.2)

    def decode_reward(symbols: list[int] | tuple[int, ...]) -> float:
        obs, rew = agent.decode_percept(list(symbols))
        return float(rew)

    _ = root_uct_action(
        xi,
        budget,
        valid_actions=env.valid_actions,
        encode_action=agent.encode_action,
        n_action_bits=env.action_bits(),
        n_percept_bits=env.percept_bits(),
        decode_reward=decode_reward,
        rng=rng,
    )

    assert math.isclose(xi.root_log_probability(), ref, rel_tol=0.0, abs_tol=1e-8)


@pyaixi
def test_root_uct_leaves_xi_root_log_p_unchanged_two_armed_bandit_wrapped() -> None:
    """Second toy (two-armed bandit + protocol adapter): ξ snapshot replay after rollouts matches CoinFlip pattern."""
    rng = random.Random(17)
    py_e = TwoArmedBandit({"arm0-p-high": 0.22, "arm1-p-high": 0.91})
    _ = PyaixiEnvAdapter(py_e)
    agent = MC_AIXI_CTW_Agent(environment=py_e, options=coin_flip_options_toy())
    tree = ctw_context_tree.CTWContextTree(depth=10)
    xi = PyAixiCTWBitMixture(tree)

    xi.learn_symbols(agent.encode_percept(py_e.observation, py_e.reward))
    ref = xi.root_log_probability()

    budget = MCTSSearchBudget(mc_simulations=14, planning_horizon=3, uct_exploration_c=1.15)

    def decode_reward(symbols: list[int] | tuple[int, ...]) -> float:
        _obs, rew = agent.decode_percept(list(symbols))
        return float(rew)

    _ = root_uct_action(
        xi,
        budget,
        valid_actions=py_e.valid_actions,
        encode_action=agent.encode_action,
        n_action_bits=py_e.action_bits(),
        n_percept_bits=py_e.percept_bits(),
        decode_reward=decode_reward,
        rng=rng,
    )

    assert math.isclose(xi.root_log_probability(), ref, rel_tol=0.0, abs_tol=1e-8)


@pyaixi
def test_mcts_mean_return_vs_baselines_biased_coin() -> None:
    """Non-binding smoke: one fixed seed block; loose margin vs baselines (not a statistical guarantee)."""
    horizon = 10
    episodes = 48
    budget = MCTSSearchBudget(mc_simulations=16, planning_horizon=4, uct_exploration_c=1.25)

    def run_block(policy: str, seed: int) -> float:
        rng = random.Random(seed)
        total = 0.0
        for ep in range(episodes):
            erng = random.Random(rng.randint(0, 2**30))
            env = CoinFlip({"coin-flip-p": 0.72})
            agent = MC_AIXI_CTW_Agent(environment=env, options=coin_flip_options_toy())
            tree = ctw_context_tree.CTWContextTree(depth=12)
            xi = PyAixiCTWBitMixture(tree)
            xi.learn_symbols(agent.encode_percept(env.observation, env.reward))

            ep_ret = 0.0
            for _t in range(horizon):
                if policy == "mcts":

                    def decode_reward(symbols: list[int] | tuple[int, ...]) -> float:
                        _obs, rew = agent.decode_percept(list(symbols))
                        return float(rew)

                    a = root_uct_action(
                        xi,
                        budget,
                        valid_actions=env.valid_actions,
                        encode_action=agent.encode_action,
                        n_action_bits=env.action_bits(),
                        n_percept_bits=env.percept_bits(),
                        decode_reward=decode_reward,
                        rng=erng,
                    )
                elif policy == "random":
                    a = erng.choice(env.valid_actions)
                else:
                    a = env.valid_actions[0]

                xi.append_history_symbols(agent.encode_action(a))
                env.perform_action(a)
                xi.learn_symbols(agent.encode_percept(env.observation, env.reward))
                ep_ret += float(env.reward)
            total += ep_ret
        return total / episodes

    m_mcts = run_block("mcts", seed=9001)
    m_rand = run_block("random", seed=9001)
    m_first = run_block("first", seed=9001)

    assert m_mcts >= m_rand - 0.25, "MCTS should not trail random by much on this toy"
    assert m_mcts >= m_first - 0.25, "MCTS should not trail first-action by much on this toy"
