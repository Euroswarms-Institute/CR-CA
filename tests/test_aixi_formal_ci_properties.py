"""
WP-Formal-CI property matrix (SWA-21 plan §10, P1–P7).

Blueprint: revert/replay, ω normalization, on-policy guardrails for ω,
augmentation determinism, CI timeout bounds. Shared ξ interface: ``aixi.formal_ci``.
"""

from __future__ import annotations

import math

import pytest

from aixi.formal_ci import (
    ToyPolicyMixture,
    ToyRevertibleMixture,
    augmentation_cycle_slot,
    augmentation_epoch_index,
    augmentation_phase_index,
    assert_predict_update_under_budget,
    epsilon_greedy_epsilon,
    omega_sum,
    run_under_wall_clock,
)

pytestmark = [
    pytest.mark.aixi_formal_ci,
]


# --- P1: revert() restores state (Families A, B) ---


@pytest.mark.aixi_p1
def test_p1_revert_restores_weights_after_update() -> None:
    m = ToyRevertibleMixture(log_weights={"a": 0.0, "b": -0.5})
    before = m.current_weights_copy()
    m.update((0, 1.0))
    assert m.current_weights_copy() != before
    m.revert()
    assert m.current_weights_copy() == before


@pytest.mark.aixi_p1
def test_p1_revert_raises_on_empty_stack() -> None:
    m = ToyRevertibleMixture()
    with pytest.raises(RuntimeError, match="empty"):
        m.revert()


# --- P2: empty stack → pre-search snapshot (Family A integration-style) ---


@pytest.mark.aixi_p2
def test_p2_revert_all_restores_baseline_after_search_simulation() -> None:
    m = ToyRevertibleMixture(log_weights={"m0": 0.0})
    baseline = m.baseline_snapshot()
    for _ in range(5):
        m.update("percept")
    m.revert_all()
    assert m.current_weights_copy() == baseline


# --- P3: sum_π ω(π|h) ≈ 1 (Family B) ---


@pytest.mark.aixi_p3
def test_p3_omega_normalizes_after_log_updates() -> None:
    p = ToyPolicyMixture(log_weights={0: 0.0, 1: -0.7, 2: -1.4})
    w = p.omega()
    assert len(w) == 3
    assert math.isclose(omega_sum(w), 1.0, rel_tol=1e-9, abs_tol=1e-12)


# --- P4: no ω update on unrealized counterfactual branches (Family B) ---


@pytest.mark.aixi_p4
def test_p4_counterfactual_clone_does_not_mutate_live_posterior() -> None:
    live = ToyPolicyMixture(log_weights={0: 0.0, 1: 0.0})
    omega0 = live.omega()
    branch = live.branch_counterfactual_clone()
    branch.update_from_observed_action(1, learning_rate=2.0)
    assert live.omega() == omega0


@pytest.mark.aixi_p4
def test_p4_simulate_counterfactual_branch_forbidden_on_live() -> None:
    live = ToyPolicyMixture(log_weights={0: 0.0})
    with pytest.raises(RuntimeError, match="branch_counterfactual_clone"):
        live.simulate_counterfactual_branch(0, 1.0)


# --- P5: augmentation index pure in (t, N) (Family C) ---


@pytest.mark.aixi_p5
@pytest.mark.parametrize(
    "t,n,expected",
    [
        (0, 10, 0),
        (9, 10, 0),
        (10, 10, 1),
        (100, 7, 14),
    ],
)
def test_p5_augmentation_phase_deterministic(t: int, n: int, expected: int) -> None:
    assert augmentation_epoch_index(t, n) == expected
    assert augmentation_phase_index(t, n) == expected


@pytest.mark.aixi_p5
def test_p5_augmentation_cycle_slot_matches_mod() -> None:
    assert augmentation_cycle_slot(0, 10) == 0
    assert augmentation_cycle_slot(9, 10) == 9
    assert augmentation_cycle_slot(10, 10) == 0
    assert augmentation_cycle_slot(100, 7) == 2


@pytest.mark.aixi_p5
def test_p5_invalid_period_rejected() -> None:
    with pytest.raises(ValueError):
        augmentation_epoch_index(0, 0)
    with pytest.raises(ValueError):
        augmentation_cycle_slot(0, 0)


# --- P6: ε schedule matches documented linear table (Family C smoke) ---


@pytest.mark.aixi_p6
def test_p6_linear_epsilon_decay_endpoints() -> None:
    assert math.isclose(
        epsilon_greedy_epsilon(0, epsilon_start=1.0, epsilon_end=0.1, decay_steps=10),
        1.0,
    )
    assert math.isclose(
        epsilon_greedy_epsilon(9, epsilon_start=1.0, epsilon_end=0.1, decay_steps=10),
        0.1,
    )
    assert math.isclose(
        epsilon_greedy_epsilon(100, epsilon_start=1.0, epsilon_end=0.1, decay_steps=10),
        0.1,
    )


@pytest.mark.aixi_p6
def test_p6_linear_interior_matches_closed_form() -> None:
    start, end, decay = 0.8, 0.2, 5
    for s in range(decay):
        t = s / (decay - 1)
        want = start + (end - start) * t
        got = epsilon_greedy_epsilon(s, epsilon_start=start, epsilon_end=end, decay_steps=decay)
        assert math.isclose(got, want, rel_tol=1e-12)


# --- P7: per-call wall-clock bound (all families — smoke) ---


@pytest.mark.aixi_p7
def test_p7_predict_update_under_loose_budget() -> None:
    m = ToyRevertibleMixture(log_weights={"m0": 0.0})
    assert_predict_update_under_budget(m, budget_s=1.0)


@pytest.mark.aixi_p7
def test_p7_run_under_wall_clock_fast_noop() -> None:
    def noop() -> None:
        return None

    run_under_wall_clock(0.5, noop)
