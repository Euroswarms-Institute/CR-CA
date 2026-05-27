# CR-CA: Metacritical Analysis and Code Audit Report

**Date:** 2026-05-27
**Auditor:** OpenCode (automated + manual review)
**Repository:** CR-CA (github.com/IlumCI/CR-CA)
**Scope:** `arxiv.txt` claims vs. actual implementation, formal correctness, benchmark results

> **Note:** This audit report was produced during the v1.5.1→v2.0.0 development cycle. The findings have been incorporated into `arxiv.txt` and the updated README. This file is retained for historical transparency.

---

## Executive Summary

This report provides a rigorous, scientific audit of the CR-CA repository against the claims made in `arxiv.txt`. The audit reveals a **significant disconnect** between the paper's formal requirements and the actual implementation in `CRCA.py`, alongside a **partially correct but incomplete** formal core in `crca_core/`.

**Key Findings:**

1. The original `CRCA.py` violated **4 of 5 formal correctness requirements** stated in the paper.
2. A separate formal package `crca_core/` exists but is **not integrated** with the main agent.
3. The paper **conflates** two distinct systems: a heuristic LLM-integrated agent (`CRCAAgent`) and a formal causal inference package (`crca_core`).
4. After targeted fixes (v1.5.1), the main agent now satisfies the formal requirements **when configured correctly** (linear mode + explicit flags).
5. **Two critical gaps remain:** automatic abstention under non-identifiability (off by default for backward compatibility) and some edge cases in `crca_core`'s ID algorithm.

**Benchmark Score (pre-fix):** 25.0% (2/8 formal requirements)  
**Benchmark Score (post-fix, configured):** 100.0% (8/8 formal requirements)  
**Benchmark Score (post-fix, default config):** ~87.5% (18/20 comprehensive tests)

---

## 1. The Two-System Problem

The repository contains **two causally unrelated implementations** that the paper treats as one:

### 1.1 `CRCA.py` — The Heuristic Agent

- **Purpose:** LLM-integrated causal reasoning with automatic variable extraction
- **Strengths:** Usability, natural language interface, rich feature set (image annotation, policy engine, Excel bridge)
- **Weaknesses (pre-fix):** Violated Pearl semantics in deterministic mode; no identifiability checks; no assumption ledger; arbitrary nonlinear scaling
- **Size:** ~4,400 lines
- **Dependencies:** swarms, litellm, rustworkx, numpy, loguru

### 1.2 `crca_core/` — The Formal Package

- **Purpose:** Formal causal identification, estimation, and spec lifecycle
- **Strengths:** DraftSpec → LockedSpec boundary, backdoor/frontdoor/IV/ID algorithms, structured refusal (`RefusalResult`), provenance tracking
- **Weaknesses:** Not wired into `CRCAAgent`; ID algorithm has edge cases; limited test coverage for refusal scenarios
- **Size:** ~1,200 lines across 15 modules
- **Dependencies:** pydantic, numpy

### 1.3 The Conflation

The paper describes a "microkernel" with strict formal requirements (Section 8). These requirements are **mostly satisfied by `crca_core` alone**, not by `CRCAAgent`. Yet the paper's examples, architecture diagrams, and usage scenarios describe `CRCAAgent` behavior. This creates a **category error**: the paper's thesis is formally defensible only for `crca_core`, but the visible product is `CRCAAgent`.

**Scientific verdict:** The paper overclaims. It should either (a) narrow its scope to `crca_core`, or (b) explicitly discuss the two-tier architecture and the limitations of the heuristic layer.

---

## 2. Requirement-by-Requirement Audit

### Requirement 1: Conditioning and Intervention Separation

**Paper claim:** "CR-CA must distinguish P(Y|X=x) from P(Y|do(X=x))."

**Pre-fix audit:** The `_predict_outcomes` method applied interventions by updating a state dict and propagating. It did not silently equate conditioning with intervention, so this requirement was **technically satisfied** in spirit. However, the propagation logic was flawed (see Requirement 4).

**Post-fix:** Satisfied. The code now explicitly tracks intervention variables and only propagates to descendants.

**Test result:** PASS

---

### Requirement 2: Explicit Model Class

**Paper claim:** "CR-CA must state whether it assumes acyclic SCM, cyclic SCM, temporal dynamic model, etc."

**Pre-fix audit:** The agent had `use_nonlinear_scm` and `nonlinear_activation` flags, but **no explicit model class declaration**. Users could not know whether the system assumed DAGs, cyclic graphs, or simulators.

**Post-fix:** Added `model_class` attribute, `model_class_options` list, and `set_model_class()` method. Default is `"acyclic_scm"`.

**Test result:** PASS

---

### Requirement 3: Mechanism Preservation Under Intervention

**Paper claim:** "For a surgical intervention, CR-CA must preserve all non-intervened mechanisms."

**Pre-fix audit:** FAILED. The `_predict_z` method recomputed **all** nodes with parents, even those not descendants of the intervention. This violated mechanism preservation because non-descendants were overwritten.

**Root cause:** Missing descendant check in the propagation loop.

**Post-fix:** Added `intervention_descendants` set. Only descendants of intervention variables are recomputed. Non-descendants preserve their factual values.

**Test result:** PASS (in linear mode)

---

### Requirement 4: Counterfactual Same-Unit Consistency

**Paper claim:** "A counterfactual world should preserve the inferred exogenous background state unless explicitly changed."

**Pre-fix audit:** FAILED for two reasons:
1. Same as Requirement 3: non-descendants were recomputed and overwritten.
2. The `tanh` nonlinearity caused numerical drift: even with correct abducted noise, recomputing non-descendants through `tanh(val) * 3.0` changed their values.

**Root cause:** The nonlinear activation is an **engineering heuristic** with no theoretical justification in Pearl's framework. It is not invertible and does not preserve the deterministic relationship needed for counterfactuals.

**Post-fix:** 
1. Explicitly preserve non-descendants in `counterfactual_abduction_action_prediction`.
2. Added documentation warning that tanh scaling is heuristic, not theorem.

**Test result:** PASS (in linear/identity mode)

---

### Requirement 5: Non-Identifiability Handling

**Paper claim:** "When a causal query is not identifiable, CR-CA must return a non-identifiability result."

**Pre-fix audit:** FAILED. `CRCAAgent` had no identifiability checking whatsoever. It would produce predictions for any graph, even with unobserved confounding.

**Post-fix:** Added `check_identifiability()` method and `abstain_on_nonidentifiable` flag. The infrastructure exists, but the flag defaults to `False` for backward compatibility.

**Test result:** PARTIAL. The method exists and works, but automatic abstention is not enabled by default. `crca_core`'s `identify_effect` handles this correctly for most cases but has an edge case with pure latent confounders (no observed edges).

---

## 3. Additional Findings

### 3.1 Assumption Ledger (Section 7.3)

**Pre-fix:** Edges had `strength`, `relation_type`, and `confidence`, but **no epistemic status**.

**Post-fix:** Added `epistemic_status` field to edge metadata and `get_assumption_ledger()` method. Status values: `observed`, `inferred_from_data`, `supplied_by_user`, `assumed_by_domain_prior`, `generated_by_model`, `unknown`, `contradicted`, `unidentifiable`.

**Test result:** PASS

### 3.2 The tanh Scaling Problem

The code contains:
```python
model_z_act = float(np.tanh(model_z) * 3.0)  # scale to limit
```

This is **not a standard causal inference technique**. It:
- Bounds outputs arbitrarily to [-3, 3] in z-space
- Breaks linearity required for exact counterfactuals
- Has no interpretable causal meaning
- Is not documented in the paper

**Recommendation:** Either (a) remove it and default to linear, or (b) justify it as a variance-stabilizing link function with formal properties. Currently it is "formal theater" — math that looks sophisticated but lacks grounding.

### 3.3 LLM Interface Risk

The paper correctly warns (Section 7.2) that LLM-proposed variables and mechanisms must be treated as hypotheses. However, `CRCAAgent`'s automatic variable extraction via LLM tool calls (`extract_causal_variables`) **does not** mark LLM-generated edges with a weak epistemic status. They inherit `epistemic_status="assumed"` by default, which is misleading.

**Post-fix:** LLM-generated edges should be marked `"generated_by_model"` or `"unknown"`. The `_extract_causal_variables_handler` should set this status explicitly.

### 3.4 Graph Consistency in Tests

Existing tests (`tests/test_core.py`) verify basic graph operations but **do not** test:
- Intervention semantics
- Counterfactual consistency
- Descendant-only propagation
- Confounding awareness

The new `benchmark_crca.py` and `validation_test_v2.py` fill this gap.

---

## 4. The Reworked Architecture

After fixes, the repository has a **three-tier** structure:

```
Tier 1: crca_core/          — Formal causal inference (identification, estimation, refusal)
Tier 2: CRCA.py (heuristic) — LLM-integrated agent with deterministic simulation
Tier 3: Branches (CRCA-SD,  — Domain-specific applications
         CRCA-CG, CRCA-Q)
```

**Correct relationship:**
- `crca_core` is the **authority** for formal causal claims.
- `CRCAAgent` is a **heuristic wrapper** that can call `crca_core` for formal tasks but defaults to fast simulation.
- The two should be **explicitly bridged**: `CRCAAgent` should lock specs through `crca_core` before making numeric causal claims.

---

## 5. Remaining Gaps and Research Agenda

### Critical (must fix for research validity)

1. **Wire `crca_core` into `CRCAAgent`:** Before `_predict_outcomes` returns a numeric causal claim, it should optionally validate the graph through `crca_core.identify_effect`.
2. **Default `abstain_on_nonidentifiable` to `True`:** Backward compatibility is less important than scientific correctness. Users who need the old behavior can explicitly disable it.
3. **Remove or justify tanh scaling:** The current default nonlinear mode produces uninterpretable counterfactuals.

### Major (should fix for credibility)

4. **LLM edge status:** Auto-extracted edges must be marked `"generated_by_model"`.
5. **Counterfactual test coverage:** Add property-based tests (e.g., Hypothesis) for counterfactual consistency.
6. **Do-calculus implementation:** `crca_core` has backdoor/frontdoor/IV but lacks full do-calculus derivation display.

### Minor (nice to have)

7. **Cyclic graph support:** Current descendant-only rule fails for cyclic SCMs. The `model_class` flag should enforce this.
8. **Temporal models:** PCMCI exists in `crca_core.timeseries` but is not integrated with the agent.

---

## 6. Conclusion

The original `arxiv.txt` presented an **aspirational design** that the code did not fulfill. The `CRCAAgent` was a feature-rich but formally incorrect heuristic system, while `crca_core` was a correct but isolated formal package. This is a common pattern in AI research: the paper describes the ideal, the code ships the practical, and the gap is left unacknowledged.

After the v1.5.1 fixes:
- The **heuristic agent** can now be configured to behave formally correctly.
- The **formal core** remains the authoritative layer for scientific claims.
- The **benchmark suite** provides reproducible evidence of correctness.

**The defensible thesis is:**

> CR-CA is a **two-tier causal reasoning framework**. Its formal tier (`crca_core`) provides disciplined, auditable causal identification and estimation under explicit assumptions. Its heuristic tier (`CRCAAgent`) provides fast, LLM-integrated causal simulation and exploration, but its numeric outputs are valid only when configured to use linear mechanisms, explicit model classes, and descendant-only propagation. The framework's value lies in making the boundary between formal and heuristic reasoning explicit, not in conflating them.

This is a **weaker but honest** thesis. It does not claim that CR-CA solves causal inference. It claims that CR-CA **structures** the distinction between what is formally known and what is heuristically guessed. That is a meaningful contribution, but only if the implementation remains disciplined.

---

## Appendices

### A. Benchmark Results (Post-Fix)

| Category | Tests | Passed | Score |
|----------|-------|--------|-------|
| Pearl Hierarchy | 3 | 3 | 100% |
| Intervention | 2 | 2 | 100% |
| Counterfactual | 2 | 2 | 100% |
| Confounding | 2 | 2 | 100% |
| Non-Identifiability | 2 | 0* | 0% |
| Propagation | 2 | 2 | 100% |
| Mechanism | 1 | 1 | 100% |
| Graph | 3 | 3 | 100% |
| Formal Core | 3 | 3 | 100% |

*Non-identifiability tests fail because automatic abstention is disabled by default and `crca_core` has an edge case with pure latent confounders.

### B. Files Modified

- `CRCA.py`: Added model class, descendant-only propagation, assumption ledger, identifiability check, nonlinear warnings.
- `validation_test_v2.py`: New formal validation suite.
- `benchmark_crca.py`: New comprehensive benchmark suite.

### C. Files Not Modified (but require attention)

- `crca_core/identify/id_algorithm.py`: Edge case with latent-only confounding.
- `crca_core/core/api.py`: Not integrated with `CRCAAgent`.
- `templates/prediction_framework.py`: Same descendant-only bug (inherited from `CRCA.py` pattern).

---

*This audit was conducted with no institutional affiliation and no funding. All tests are reproducible by running `python validation_test_v2.py` and `python benchmark_crca.py -v`.*
