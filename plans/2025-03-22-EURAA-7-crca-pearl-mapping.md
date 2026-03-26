# EURAA-7: CRCA ↔ Pearl Mapping — Alignments and Divergences

**Task:** Mapping document between CRCA.py constructs and Pearl's causal framework  
**Author:** Research Engineer  
**Date:** 2025-03-22  
**References:** plans/2025-03-22-EURAA-5-research-plan.md, plans/2025-03-22-EURAA-6-pearl-synthesis.md, CRCA.py, crca_core

---

## 1. Summary (200 words)

This document maps CRCA.py's causal constructs (causal graph, interventions, counterfactual scenarios, SCM simulation) to Judea Pearl's framework. CRCA has **two layers**: (1) **CRCA.py** — an LLM-integrated, graph-based agent with heuristic SCM-style propagation, nonlinear activations, and meta-Monte Carlo counterfactual sampling; (2) **crca_core** — a research-grade, refusal-first core implementing Pearl-aligned identification, estimation, and counterfactuals via explicit linear-Gaussian SCMs.

**Alignments:** crca_core closely follows Pearl: SCM abduction–action–prediction, backdoor/frontdoor/IV identification, do-calculus semantics. CRCA.py's graph surgery and topological propagation approximate do-interventions; `identify_adjustment_set` approximates backdoor logic.

**Divergences:** CRCA.py uses a graph with edge strengths, confidence, standardization, optional tanh nonlinearity, and `shock_preserve_threshold`—none of which are in Pearl's formal SCM semantics. Its counterfactual "abduction" reuses residuals from the factual z-state rather than inferring exogenous U from a true structural model. Pearl requires a full SCM for Level 3; CRCA.py can run without one. crca_core refuses counterfactuals without an explicit SCM; CRCA.py does not.

---

## 2. Construct-by-Construct Mapping

### 2.1 Causal Graph

| CRCA construct | Pearl analogue | Alignment | Divergence |
|---------------|----------------|-----------|------------|
| **CRCA.py** `causal_graph` (dict + rustworkx PyDiGraph) | Causal DAG G | ✓ Directed edges; nodes = variables | CRCA stores `strength`, `confidence`, `relation_type` per edge; Pearl's DAG has no edge weights. |
| **crca_core** `CausalGraphSpec` / `CausalGraph` (from_spec) | Causal DAG G with optional latent confounders | ✓ Nodes, directed edges, `latent_confounders` (bidirected) | crca_core supports bidirected edges; CRCA.py does not model latent confounding explicitly. |
| CRCA.py `CausalNode`, `CausalEdge`, `CausalRelationType` | N/A | — | Pearl does not use relation types (direct/indirect/confounding/mediating/moderating) as first-class graph annotations. |
| crca_core `d_separated`, `remove_outgoing`, `remove_incoming` | d-separation, graph surgery for do-calculus | ✓ | Aligned. |

**Verdict:** crca_core's graph is Pearl-aligned. CRCA.py's graph extends Pearl with numeric metadata; topology is compatible, semantics of strength/confidence are not in Pearl.

---

### 2.2 Interventions (do-operator)

| CRCA construct | Pearl analogue | Alignment | Divergence |
|----------------|----------------|-----------|------------|
| **CRCA.py** `CounterfactualScenario.interventions` | \(do(X=x)\) | ✓ Graph surgery: intervened variables set directly, parents ignored | CRCA.py applies interventions in z-space (standardized); Pearl's do is on raw variables. |
| **CRCA.py** `_predict_z` / `_predict_outcomes` | \(P(Y \mid do(X=x))\) | ✓ Topological order; intervened nodes bypass parent contribution | CRCA.py uses `shock_preserve_threshold` to optionally preserve observed values for "shocked" nodes—non-Pearl heuristic. |
| **crca_core** `simulate_counterfactual` intervention dict | \(do(X=x)\) | ✓ Full Pearl semantics: structural equations replaced for intervened vars | Aligned. |
| **crca_core** `design_intervention` | Experiment design for identification | ✓ Suggests RCT, confounder measurement, IV designs | Graphical, non-probabilistic; does not compute \(P(Y\mid do(X))\). |

**Verdict:** crca_core's intervention semantics match Pearl. CRCA.py's propagation approximates do-interventions but adds heuristics (standardization, shock preservation) not in Pearl.

---

### 2.3 Counterfactual Scenarios

| CRCA construct | Pearl analogue | Alignment | Divergence |
|----------------|----------------|-----------|------------|
| **Pearl Level 3** | \(P(Y_x = y \mid X=x', Y=y')\) | — | Requires: (1) factual observation, (2) abduce U, (3) intervene, (4) predict. |
| **crca_core** `simulate_counterfactual` | ✓ Abduction–action–prediction | ✓ Full alignment | `LinearGaussianSCM.abduce_noise` → `predict(u, interventions)`. Refuses if no SCM. |
| **CRCA.py** `counterfactual_abduction_action_prediction` | Partial alignment | ~ | Uses residuals \(z_i - \sum_j \beta_j z_j\) as "noise"; propagates with same residuals under intervention. **Issue:** These are not exogenous U in Pearl's sense—they conflate measurement error and structural noise. |
| **CRCA.py** `generate_counterfactual_scenarios` (Monte Carlo) | N/A | — | Pearl does not define "scenario generation"; this samples interventions and graph variants for exploratory analysis. Heuristic, not formal counterfactual inference. |

**Key divergence:** Pearl's counterfactual requires a **fully specified SCM** (structural equations + noise model). CRCA.py runs without one: it treats graph edge strengths as coefficients and residualizes from the factual state. That yields *something like* a linear SCM, but:
- No explicit noise model (U is implicit in residuals)
- Standardization and nonlinearities (tanh) break Pearl's linear-Gaussian assumptions
- `identify_adjustment_set` in CRCA.py uses parents-of-treatment only—not the full backdoor criterion

**Verdict:** crca_core implements Pearl's Level 3. CRCA.py implements a heuristic approximation that may coincide when the graph is linear, standardized, and the "abduction" is well-defined.

---

### 2.4 SCM Simulation

| CRCA construct | Pearl analogue | Alignment | Divergence |
|----------------|----------------|-----------|------------|
| **Pearl SCM** | \(X_i = f_i(\text{Pa}(X_i), U_i)\) | — | Structural equations + exogenous U. |
| **crca_core** `LinearGaussianSCM` | ✓ Linear-Gaussian SCM | ✓ \(V_i = \beta_0 + \sum_j \beta_j V_j + U_i\), U independent | Aligned. Supports `abduce_noise`, `predict`, `counterfactual`. |
| **CRCA.py** `_predict_z` | Approximate linear/nonlinear SCM | ~ | \(z_i = \tanh(\sum_j \beta_j z_j + \text{noise}) \cdot 3\) or linear. Noise = \(z_i - \sum_j \beta_j z_j\) (residual). **No explicit U distribution.** |
| **CRCA.py** `standardization_stats` | N/A | — | Pearl typically works in original scale or assumes known distributions. CRCA.py standardizes for numerical stability. |
| **CRCA.py** `interaction_terms` | Nonlinear SCM | — | Pearl's basic SCM is additive; CRCA.py allows \(z_1 \cdot z_2\) interaction terms—extensible but not standard. |

**Verdict:** crca_core's SCM is Pearl-compliant. CRCA.py's simulation is a functional approximation with ad hoc choices (standardization, tanh, residual-as-noise).

---

## 3. Three-Level Hierarchy Mapping

| Pearl level | CRCA.py | crca_core |
|-------------|---------|-----------|
| **L1 Association** | `_predict_outcomes` with no intervention (observe) | Not primary; `estimate_effect_dowhy` uses observational data for identification/estimation. |
| **L2 Intervention** | `_predict_outcomes(factual, interventions)` | `identify_effect` + `estimate_effect_dowhy`; `design_intervention` for experiments. |
| **L3 Counterfactual** | `counterfactual_abduction_action_prediction` | `simulate_counterfactual` (requires SCM). |

**Note:** CRCA.py conflates L2 and L3: it uses factual-state residuals for "counterfactual" prediction. True L3 needs abduction of U from the factual world; CRCA.py's residualization is a shortcut that matches only under linear, homoskedastic assumptions.

---

## 4. Identification

| CRCA construct | Pearl analogue | Alignment |
|----------------|----------------|-----------|
| **crca_core** `identify_effect` (backdoor, frontdoor, IV, ID) | do-calculus identification | ✓ Backdoor formula \(\sum_z P(Y|X,z)P(z)\); frontdoor; IV; ID recursion. |
| **CRCA.py** `identify_adjustment_set` | Backdoor criterion | ~ Uses parents of treatment excluding descendants; not full backdoor (e.g., may miss collider-strategy cases). |
| **CRCA.py** `detect_confounders` | Common causes | ~ Ancestor intersection; heuristic. |

**Verdict:** crca_core follows Pearl's identification machinery. CRCA.py provides simplified, approximate utilities.

---

## 5. Divergence Summary Table

| Area | CRCA.py vs Pearl | crca_core vs Pearl |
|------|------------------|---------------------|
| Graph | + strength, confidence, relation type; no bidirected | Aligned; supports latent confounders |
| Intervention | Approximate; z-space, shock preservation | Aligned |
| Counterfactual | Heuristic (residual-as-noise); no SCM required | Aligned; requires SCM |
| SCM | Implicit; residual-based; optional nonlinearity | Explicit linear-Gaussian |
| Identification | Approximate backdoor | Full backdoor/frontdoor/IV/ID |

---

## 6. Scientific Analysis of This Mapping

### 200-Word Summary

This mapping document compares CRCA.py and crca_core to Pearl's causal framework. It is a **technical alignment analysis**, not original research. The document correctly distinguishes two CRCA layers and maps each construct (graph, intervention, counterfactual, SCM) to Pearl's semantics. Key findings: crca_core is closely Pearl-aligned; CRCA.py uses heuristic approximations that may work in practice but deviate from formal SCM semantics (e.g., residual-as-noise, no explicit U, standardization).

### Scientific Validity

- **~85%:** The mapping accurately reflects both CRCA implementation and Pearl's framework as summarized in EURAA-6. Construct-by-construct comparison is consistent with the code.
- **~10%:** Some nuance may be lost (e.g., partial counterfactuals, transportability).
- **~5%:** Risk of mischaracterizing edge cases in CRCA.py's propagation logic.

### Scientific Novelty

- **~0%:** No novel scientific claims. This is documentation and alignment analysis.
- **~95%:** Correctly framed as mapping/reference material.

### Classification

- **Engineering documentation:** ~80%
- **Science-adjacent (applied causal inference):** ~20%

### Math Correctness

- **~90%:** Descriptions of Pearl's semantics and crca_core's implementation are correct. CRCA.py's residual-based "abduction" is correctly flagged as non-Pearl.
- **~10%:** Formula notation (e.g., backdoor) is standard; no formal proof.

### Novel vs Illusion

- **~95%:** This document does not claim novelty. It is an alignment study.
- **~5%:** None.

### Hypothesis Probabilities

| Hypothesis | P |
|------------|---|
| Mapping is factually accurate | 0.85 |
| CRCA.py divergence analysis is correct | 0.88 |
| crca_core alignment claim is correct | 0.92 |
| Document is appropriate for downstream math audit (EURAA-8) | 0.90 |

---

## 7. Email to Author

**To:** Research Engineer (self)  
**Re:** Scientific analysis of EURAA-7 CRCA ↔ Pearl mapping

The mapping document you produced for EURAA-7 is a **competent technical alignment analysis**. It correctly separates CRCA.py (heuristic, LLM-integrated) from crca_core (Pearl-aligned, refusal-first) and maps each to Pearl's framework.

**Strengths:** Clear structure, construct-by-construct comparison, explicit divergence callouts. The residual-as-noise vs exogenous-U distinction is correctly emphasized. Suitable as reference for EURAA-8 (math audit).

**Caveats:** (1) Consider adding a short "Recommendations" section: when to use crca_core vs CRCA.py for Pearl-compliant inference. (2) The `identify_adjustment_set` in CRCA.py may warrant a code-level audit to confirm it matches "parents of X \ descendants of X" in all cases. (3) For completeness, mention that CRCA.py's `identify_adjustment_set` does not use d-separation—it uses a simpler ancestor-based heuristic.

**Verdict:** Scientifically valid as documentation. Not novel. Math and semantic descriptions appear correct. Not an instance of illusory novelty—appropriately framed as alignment/reference work.

---

*Document applies the Scientific Analysis Discipline per AGENTS.md.*
