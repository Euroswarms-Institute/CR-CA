"""AIQI Phase 4 skeleton: augmentation schedule + τ-greedy + env protocol (IMPLEMENTATION_PLAN §4.3)."""

from __future__ import annotations

from random import Random

from aixi.aixi.env.toy_coin_bit import ToyBiasedCoinEnv
from aixi.aixi.planning.aiqi import (
    AIQIConfig,
    AIQISkeletonAgent,
    TabularPhaseReturnPredictor,
    UniformReturnPredictor,
    augmentation_insert_at_timestep,
    epsilon_greedy_action,
    grid_bin_value,
    mean_return_from_probs,
    total_return_to_bin,
)


def test_augmentation_schedule_matches_mod_pattern() -> None:
    n = 7
    assert augmentation_insert_at_timestep(0, 0, n) is True
    assert augmentation_insert_at_timestep(0, 1, n) is False
    assert augmentation_insert_at_timestep(14, 0, n) is True
    assert augmentation_insert_at_timestep(15, 1, n) is True
    assert augmentation_insert_at_timestep(15, 2, n) is False


def test_total_return_to_bin_normalized() -> None:
    assert total_return_to_bin(0.0, horizon_H=1, grid_bins_M=4, reward_min=0.0, reward_max=1.0) == 0
    assert total_return_to_bin(1.0, horizon_H=1, grid_bins_M=4, reward_min=0.0, reward_max=1.0) == 3


def test_tabular_phase_predictor_updates_on_policy() -> None:
    pred = TabularPhaseReturnPredictor(pseudocount=1.0)
    m = 4
    before = mean_return_from_probs(
        pred.predicted_return_probs(
            timestep=1,
            history_len=1,
            observation=0,
            action=0,
            grid_bins_M=m,
            phase_n=0,
        ),
        m,
    )
    pred.observe_augmented_return(
        phase_n=0,
        observation=0,
        action=0,
        return_bin=m - 1,
        grid_bins_M=m,
    )
    pred.observe_augmented_return(
        phase_n=0,
        observation=0,
        action=0,
        return_bin=m - 1,
        grid_bins_M=m,
    )
    after = mean_return_from_probs(
        pred.predicted_return_probs(
            timestep=1,
            history_len=1,
            observation=0,
            action=0,
            grid_bins_M=m,
            phase_n=0,
        ),
        m,
    )
    assert after > before


def test_grid_bin_and_mean_uniform_probs() -> None:
    m = 5
    assert grid_bin_value(0, m) == 0.0
    assert grid_bin_value(m - 1, m) == (m - 1) / m
    u = [0.2] * m
    expected = sum(k / m * 0.2 for k in range(m))
    assert abs(mean_return_from_probs(u, m) - expected) < 1e-9


def test_epsilon_greedy_only_valid_actions() -> None:
    rng = Random(42)
    q = {0: 0.0, 1: 5.0, 2: -1.0}
    valid = (0, 1, 2)
    for _ in range(100):
        a = epsilon_greedy_action(q, valid, exploration_tau=0.4, rng=rng)
        assert a in valid


def test_aiqi_skeleton_loop_toy_env() -> None:
    rng = Random(123)
    env = ToyBiasedCoinEnv(p_heads=0.5, rng=Random(456))
    cfg = AIQIConfig(
        horizon_H=2,
        grid_bins_M=4,
        augmentation_period_N=3,
        exploration_tau=0.2,
    )
    agent = AIQISkeletonAgent(
        env=env,
        config=cfg,
        predictor=UniformReturnPredictor(),
        rng=rng,
    )
    p = env.reset()
    agent.observe(p)
    for _ in range(8):
        a = agent.act()
        assert 0 <= a < env.action_space.size
        p2 = env.step(a)
        agent.observe(p2)
