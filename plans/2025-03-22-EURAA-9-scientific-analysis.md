# EURAA-9: Scientific Analysis — Pearl-Aligned Causal Reasoning Implementation

**Task:** Implement Pearl-aligned causal & counterfactual reasoning in CR-CA  
**Author:** Research Engineer  
**Date:** 2025-03-22

---

## 1. 200-Word Summary

This implementation fixes CRCA.py and templates/prediction_framework.py to align with Judea Pearl's causal framework, per EURAA-8 findings. **Changes made:** (1) Removed `shock_preserve_threshold`—the heuristic that preserved observed values for non-intervened nodes when |z| > 1e-3, which violated do-operator semantics. (2) Added optional `intercepts` (β₀ per node) for linear-Gaussian alignment with Pearl's V_i = β₀ + Σ β_j V_j + U_i. (3) Unified counterfactual and predict paths: `counterfactual_abduction_action_prediction` now applies the same tanh nonlinearity as `predict_outcomes` when `use_nonlinear_scm=True`, resolving the previous inconsistency. (4) Added interaction-term support to the counterfactual path for full consistency with predict. (5) Documented crca_core vs CRCA.py usage: use crca_core for Pearl-compliant identification/estimation; use CRCA.py for LLM-integrated heuristic exploration.

**Scientific validity:** The changes implement the EURAA-8 recommendations directly. No new mathematical claims are made.

---

## 2. Scientific Validity

- **~92%:** Code changes correctly remove shock_preserve, add intercepts, and unify tanh. Implementation matches audit recommendations.
- **~5%:** Interaction terms in counterfactual may have edge cases (e.g., graph_manager vs causal_graph structure differences).
- **~3%:** No empirical validation against crca_core on shared graphs.

---

## 3. Scientific Novelty

- **~0%:** This is implementation work, not novel research.
- **~98%:** Appropriately framed as alignment of existing code to known semantics.

---

## 4. Classification

- **Engineering / Technical debt remediation:** ~90%
- **Science-adjacent (causal inference tooling):** ~10%

---

## 5. Math Correctness

- **~95%:** Abduction formula U_i = Z_i - (β₀ + Σ β_j Z_j + interaction) is correct. Prediction Z_i = β₀ + Σ β_j parent_z + U_i + interaction is correct. Tanh applied consistently in both paths.
- **~5%:** Interaction-term edge-data access (interaction_strength) may vary by graph backend.

---

## 6. Novel vs Illusion

- **~98%:** No false novelty claims. Changes are explicit alignment fixes.
- **~2%:** None.

---

## 7. Hypothesis Probabilities

| Hypothesis | P |
|------------|---|
| shock_preserve removal improves Pearl compliance | 0.96 |
| Intercept support enables linear-Gaussian alignment | 0.94 |
| Tanh unification fixes counterfactual/predict inconsistency | 0.95 |
| Implementation matches EURAA-8 recommendations | 0.93 |
| crca_core remains preferred for full Pearl compliance | 0.92 |

---

## 8. Email to Author

**To:** Research Engineer (self)  
**Re:** Scientific analysis of EURAA-9 implementation

The implementation you produced for EURAA-9 **correctly applies** the EURAA-8 math audit recommendations. shock_preserve_threshold has been removed, optional intercepts added, and counterfactual/predict paths unified for tanh and interaction terms. The CRCA.py module docstring and PredictionFramework __init__ docstring now clarify when to use crca_core vs CRCA.py.

**Strengths:** Direct mapping from audit findings to code changes; no overreach; deterministic SCM core preserved per board directive.

**Caveats:** (1) Consider adding a unit test that compares CRCA.py propagation to crca_core.LinearGaussianSCM on a shared graph when both use linear mode. (2) The intercepts dict is optional and default-empty; fitting intercepts from data is out of scope.

**Verdict:** Scientifically valid as implementation work. Not novel. Math is correct. Appropriately framed; not illusory.

---

*Document applies the Scientific Analysis Discipline per AGENTS.md.*
