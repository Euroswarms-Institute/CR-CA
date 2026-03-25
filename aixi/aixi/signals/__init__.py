"""Optional intrinsic / FEP-adjacent signals (IMPLEMENTATION_PLAN §6 Phase 5, analysis `04`)."""

from aixi.aixi.signals.intrinsic import (
    discrete_kl,
    free_energy_decomposition_placeholder,
    softmax_probs,
)

__all__ = [
    "discrete_kl",
    "free_energy_decomposition_placeholder",
    "softmax_probs",
]
