"""Phase 5 optional signals: FEP-adjacent utilities, joint predictor protocol, quantum toy (plan §6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aixi.aixi.models.joint_predictor import JointSequencePredictor
from aixi.aixi.signals.intrinsic import (
    adjust_q_by_log_ratio,
    discrete_kl,
    free_energy_decomposition_placeholder,
    softmax_probs,
)
from aixi.experiments.quantum_toy import SingleQubitMeasureEnv


def test_softmax_probs_normalize() -> None:
    p = softmax_probs({0: 1.0, 1: 3.0}, (0, 1), temperature=1.0)
    assert math.isclose(sum(p.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert p[1] > p[0]


def test_discrete_kl_nonnegative_identical() -> None:
    u = {0: 0.5, 1: 0.5}
    assert discrete_kl(u, u, (0, 1)) == pytest.approx(0.0, abs=1e-9)


def test_adjust_q_by_log_ratio_noop_when_lambda_zero() -> None:
    q = {0: 1.0, 1: 2.0}
    pi = {0: 0.5, 1: 0.5}
    z = {0: 0.5, 1: 0.5}
    adjust_q_by_log_ratio(q, (0, 1), pi_star=pi, zeta=z, lam=0.0)
    assert q == {0: 1.0, 1: 2.0}


def test_free_energy_placeholder() -> None:
    assert free_energy_decomposition_placeholder(0.3, 0.7) == pytest.approx(1.0)


def test_joint_sequence_predictor_protocol() -> None:
    class _Stub:
        def log_prob_next_pair(self, history_key: object, action: int, percept_key: object) -> float:
            return -1.0 if action == 0 else -2.0

    stub: JointSequencePredictor = _Stub()  # type: ignore[assignment]
    assert isinstance(stub, JointSequencePredictor)
    assert stub.log_prob_next_pair(None, 0, "e") == -1.0


def test_single_qubit_measure_env_trace_and_rewards() -> None:
    env = SingleQubitMeasureEnv(rng=np.random.default_rng(0))
    p0 = env.reset()
    assert p0.observation == 0
    assert p0.reward == 0.0
    for _ in range(20):
        p = env.step(0)
        assert p.observation in (0, 1)
        assert p.reward == float(p.observation)
