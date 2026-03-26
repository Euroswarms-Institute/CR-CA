"""
Finite supertask surrogate (SWA-25): schedule envelope + Hypothesis properties.

Rejects hypercomputer-style public APIs by name; proves halting under caps.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aixi import supertask_surrogate as st_surrogate
from aixi.formal_ci import run_under_wall_clock
from aixi.supertask_surrogate import (
    FiniteBudgetExceeded,
    ScheduleEnvelope,
    SupertaskSurrogateState,
    assert_no_hypercomputer_api_on_module,
    drain_envelope,
    run_bounded,
)

pytestmark = [pytest.mark.aixi_supertask]


def test_module_exposes_no_hypercomputer_style_names() -> None:
    assert_no_hypercomputer_api_on_module(st_surrogate)


@settings(max_examples=200, deadline=None)
@given(
    max_macro=st.integers(min_value=0, max_value=40),
    micro_per=st.integers(min_value=1, max_value=15),
)
def test_property_drain_respects_envelope(max_macro: int, micro_per: int) -> None:
    env = ScheduleEnvelope(max_macro_steps=max_macro, micro_steps_per_macro=micro_per)
    state = SupertaskSurrogateState(envelope=env)
    used = drain_envelope(state)
    assert used == env.max_micro_steps
    assert state.micro_steps_used == env.max_micro_steps
    assert state.halted()
    assert state.micro_steps_used <= env.max_micro_steps


@settings(max_examples=200, deadline=None)
@given(
    max_macro=st.integers(min_value=1, max_value=30),
    micro_per=st.integers(min_value=1, max_value=12),
    chunk=st.integers(min_value=0, max_value=50),
)
def test_property_run_bounded_never_exceeds_request_or_envelope(
    max_macro: int, micro_per: int, chunk: int,
) -> None:
    env = ScheduleEnvelope(max_macro_steps=max_macro, micro_steps_per_macro=micro_per)
    state = SupertaskSurrogateState(envelope=env)
    cap = env.max_micro_steps
    want = min(chunk, cap)
    got = run_bounded(state, max_steps=want)
    assert got == want
    assert state.micro_steps_used == want
    assert state.micro_steps_used <= cap


@settings(max_examples=150, deadline=None)
@given(
    max_macro=st.integers(min_value=0, max_value=25),
    micro_per=st.integers(min_value=1, max_value=10),
)
def test_property_lamp_parity_after_full_run(max_macro: int, micro_per: int) -> None:
    env = ScheduleEnvelope(max_macro_steps=max_macro, micro_steps_per_macro=micro_per)
    state = SupertaskSurrogateState(envelope=env)
    drain_envelope(state)
    assert state.lamp_on == (bool(env.max_micro_steps % 2))


def test_run_bounded_rejects_over_envelope() -> None:
    env = ScheduleEnvelope(max_macro_steps=2, micro_steps_per_macro=2)
    state = SupertaskSurrogateState(envelope=env)
    with pytest.raises(FiniteBudgetExceeded):
        run_bounded(state, max_steps=5)


def test_p7_aligned_wall_clock_trivial_step() -> None:
    env = ScheduleEnvelope(max_macro_steps=3, micro_steps_per_macro=2)
    state = SupertaskSurrogateState(envelope=env)

    def body() -> None:
        drain_envelope(state)

    run_under_wall_clock(0.25, body)
    assert state.halted()


def test_zero_macro_envelope_halts_without_steps() -> None:
    env = ScheduleEnvelope(max_macro_steps=0, micro_steps_per_macro=1)
    state = SupertaskSurrogateState(envelope=env)
    state.step_micro()
    assert state.halted()
    assert state.micro_steps_used == 0
