# EURAA-8: Math Audit — CRCA SCM and Intervention Propagation vs Pearl Semantics

**Task:** Check CRCA math (linear/nonlinear SCM, intervention propagation, graph surgery) against Pearl semantics. Document any errors or deviations. Apply scientific-analysis prompt.  
**Author:** Research Engineer  
**Date:** 2025-03-22  
**References:** plans/2025-03-22-EURAA-6-pearl-synthesis.md, plans/2025-03-22-EURAA-7-crca-pearl-mapping.md, CRCA.py, templates/prediction_framework.py, crca_core

---

## 1. 200-Word Summary

This audit verifies CRCA.py and crca_core mathematical correctness against Pearl’s causal framework. **crca_core** is Pearl-aligned: LinearGaussianSCM implements proper abduction–action–prediction, graph surgery (remove_outgoing) for do-interventions, and d-separation–based backdoor identification. **CRCA.py** approximates Pearl with several deviations: (1) **shock_preserve_threshold** preserves observed values for non-intervened nodes when |z| > 1e-3, violating do-operator semantics; (2) no intercepts in SCM propagation (z_i = Σ β_j z_j vs Pearl’s V_i = β_0 + Σ β_j V_j + U_i); (3) optional tanh nonlinearity and *3 scaling are ad hoc; (4) counterfactual_abduction uses residual-as-noise (correct under linear homoskedastic assumptions) but never applies tanh—inconsistent with predict_outcomes when use_nonlinear_scm=True; (5) identify_adjustment_set uses parents-of-treatment heuristic rather than d-separation. No outright mathematical errors were found; the issues are semantic deviations and design choices that weaken Pearl alignment.

---

## 2. SCM (Structural Causal Model) Audit

### 2.1 Pearl Semantics

Pearl’s SCM: \(X_i = f_i(\text{Pa}(X_i), U_i)\), where U_i are exogenous, independent noise. Linear-Gaussian: \(V_i = \beta_0 + \sum_j \beta_j V_j + U_i\).

### 2.2 crca_core LinearGaussianSCM

| Component | Formula / behavior | Pearl alignment |
|-----------|-------------------|-----------------|
| Structural equation | \(V_i = \beta_0 + \sum_j \beta_j V_j + U_i\) | ✓ Correct |
| abduce_noise | \(U_i = x_i - (\beta_0 + \sum_j \beta_j x_j)\) | ✓ Correct |
| predict (with interventions) | Replace intervened vars; propagate in topo order | ✓ Graph surgery |
| counterfactual | u = abduce_noise(factual); predict(u, interventions) | ✓ Abduction–action–prediction |

**Verdict:** crca_core SCM is mathematically correct and Pearl-aligned.

### 2.3 CRCA.py SCM

| Component | Formula / behavior | Pearl alignment |
|-----------|-------------------|-----------------|
| Linear propagation | \(z_i = \sum_j \beta_j z_j\) | ~ No intercept |
| Nonlinear (tanh) | \(z_i = 3 \cdot \tanh(\sum_j \beta_j z_j + \gamma z_1 z_2 + \text{noise})\) | ✗ Ad hoc |
| Intercepts | Not used in propagation | ✗ Pearl uses β_0 |
| counterfactual_abduction | noise = z − Σ β_j z_j; predict with same noise | ~ Correct for linear; no tanh |

**Findings:**

- **Missing intercept:** CRCA.py uses \(z_i = \sum_j \beta_j z_j\) with no \(\beta_0\). Under standardization (mean ≈ 0), this is often acceptable, but it is a deviation from Pearl’s general form.
- **tanh * 3:** The nonlinear path uses \(\tanh(\cdot) \times 3\). This is not derived from Pearl; it is a bounded-activation design choice. It breaks linear-Gaussian assumptions.
- **Residual-as-noise:** Under a linear homoskedastic model, \(U_i = Z_i - \sum_j \beta_j Z_j\) is the exogenous noise. CRCA.py’s residualization is correct in that setting. When tanh is used in factual prediction but not in counterfactual, the two paths are inconsistent.

---

## 3. Intervention Propagation and Graph Surgery Audit

### 3.1 Pearl do-Operator

\(do(X=x)\): Replace the structural equation for X; X no longer depends on its parents. Propagate in topological order.

### 3.2 CRCA.py Implementation

**Correct behavior (lines 204–206 in prediction_framework.py):**

```python
if node in interventions:
    z_pred[node] = z_state.get(node, 0.0)
    continue
```

Intervened nodes get their value directly and skip parent contribution. Graph surgery semantics are correct.

**Deviation: shock_preserve_threshold (lines 243–247):**

```python
threshold = float(getattr(self, "shock_preserve_threshold", 1e-3))
if abs(observed_z) > threshold:
    z_pred[node] = float(observed_z)
else:
    z_pred[node] = float(model_z_act)
```

For **non-intervened** nodes, when |observed_z| > 1e-3, the code keeps the observed value instead of the model prediction. This mixes observation with intervention: under \(do(X=x)\), downstream variables should be recomputed from the intervened state, not frozen to the factual observation. This breaks Pearl’s do-operator semantics.

**Severity:** High for formal Pearl compliance. Functionally it may behave like a “preserve large deviations” heuristic, but it is not a valid causal intervention.

### 3.3 crca_core

Uses standard graph surgery only: intervened variables are set and propagation follows structural equations. No shock-preserve heuristic. ✓ Correct.

---

## 4. Identification Audit

### 4.1 Pearl Backdoor Criterion

Z is a valid adjustment set if:
1. No element of Z is a descendant of X.
2. Z blocks all backdoor paths from X to Y.

For a DAG, \(\text{Pa}(X) \setminus \text{Desc}(X) \setminus \{Y\}\) satisfies the backdoor criterion in typical cases.

### 4.2 CRCA.py identify_adjustment_set

```python
parents_t = set(self._get_parents(treatment))
descendants_t = set(self._get_descendants(treatment))
adjustment = [z for z in parents_t if z not in descendants_t and z != outcome]
```

This returns \(\text{Pa}(X) \setminus \text{Desc}(X) \setminus \{Y\}\), which is a valid backdoor set when the parents satisfy the criterion. However:

- No d-separation check.
- No latent confounders (crca_core supports bidirected edges).
- No frontdoor or IV logic.

**Verdict:** Simplified, often valid heuristic; not a full Pearl identification engine.

### 4.3 crca_core Backdoor

Uses `remove_outgoing([x])` and `d_separated([x], [y], z)` for validation. ✓ Correct Pearl alignment.

---

## 5. Counterfactual Consistency

| Method | CRCA.py | crca_core |
|--------|---------|-----------|
| Abduction | Residual z − Σ β_j z_j | U = V − (β_0 + Σ β_j V_j) |
| Action | Graph surgery (intervention overwrite) | Graph surgery |
| Prediction | Linear only (no tanh in counterfactual path) | Linear only |
| Consistency | Counterfactual path ignores tanh used in predict_outcomes | N/A (always linear) |

**Finding:** When `use_nonlinear_scm=True`, `predict_outcomes` uses tanh, but `counterfactual_abduction_action_prediction` does not. The counterfactual path is always linear. This is an internal inconsistency. When `use_nonlinear_scm=False`, both paths are linear and consistent.

---

## 6. Error Summary Table

| Area | Type | Severity | Description |
|------|------|----------|-------------|
| shock_preserve_threshold | Semantic deviation | High | Preserves observed values for non-intervened nodes; violates do-operator |
| Missing intercept | Design deviation | Medium | SCM has no β_0; acceptable under standardization |
| tanh in SCM | Design choice | Medium | Non-Pearl nonlinearity; ad hoc |
| Counterfactual vs predict | Inconsistency | Medium | Counterfactual always linear; predict can use tanh |
| identify_adjustment_set | Simplified | Low | Parents heuristic; no d-separation |

**No outright mathematical errors** (e.g. wrong formulas, sign errors) were found. The problems are semantic and design deviations from Pearl.

---

## 7. Scientific Analysis

### 7.1 Scientific Validity

- **~88%:** The audit correctly traces CRCA.py and crca_core against Pearl. The shock_preserve_threshold and missing-intercept findings are supported by code inspection.
- **~8%:** Some corner cases (e.g. colliders, unobserved variables) may need further checking.
- **~4%:** Possible oversights in graph variants or batch prediction paths.

### 7.2 Scientific Novelty

- **~0%:** This is an audit, not novel research.
- **~95%:** Appropriately framed as documentation/verification.

### 7.3 Classification

- **Technical audit / documentation:** ~85%
- **Science-adjacent (causal inference):** ~15%

### 7.4 Math Correctness

- **~92%:** Descriptions of Pearl semantics and crca_core implementation are correct. CRCA.py deviations are accurately characterized.
- **~8%:** No formal proof of backdoor sufficiency in all DAG topologies; relies on standard literature.

### 7.5 Novel vs Illusion

- **~95%:** No false novelty claims; this is an alignment audit.
- **~5%:** None.

### 7.6 Hypothesis Probabilities

| Hypothesis | P |
|------------|---|
| shock_preserve_threshold violates Pearl semantics | 0.95 |
| crca_core is Pearl-aligned | 0.93 |
| CRCA.py identify_adjustment_set is a valid but simplified backdoor | 0.88 |
| Counterfactual vs predict inconsistency exists | 0.90 |
| No mathematical formula errors in propagation | 0.92 |

---

## 8. Recommendations

1. **Remove or document shock_preserve_threshold:** Either remove it for Pearl compliance, or clearly document it as a non-causal heuristic and disable it by default when causal semantics are required.
2. **Add optional intercepts:** Support \(\beta_0\) in CRCA.py SCM propagation for full linear-Gaussian alignment.
3. **Unify counterfactual and predict:** Either apply tanh consistently in both paths or document that counterfactuals assume a linear SCM.
4. **Clarify when to use crca_core:** Use crca_core for Pearl-compliant identification, estimation, and counterfactuals; use CRCA.py for heuristic, LLM-integrated exploration.

---

## 9. Email to Author

**To:** Research Engineer (self)  
**Re:** Scientific analysis of EURAA-8 math audit

The math audit you produced for EURAA-8 is a **thorough technical verification** of CRCA.py and crca_core against Pearl’s causal framework. It correctly separates Pearl-aligned components (crca_core) from heuristic approximations (CRCA.py) and documents specific deviations.

**Strengths:** Clear structure, code-backed findings, explicit severity ratings, and actionable recommendations. The shock_preserve_threshold issue is correctly identified as a major semantic violation.

**Caveats:** (1) Consider running a small regression test comparing CRCA.py propagation to crca_core on a shared graph to quantify numeric divergence. (2) The identify_adjustment_set behavior could be compared directly to crca_core’s find_backdoor_adjustment_set on the same graphs. (3) For completeness, note that CRCA.py’s linear path (when use_nonlinear_scm=False) has no tanh—so the counterfactual/predict inconsistency only arises when nonlinear is enabled.

**Verdict:** Scientifically valid as an audit. Not novel. Math and semantic descriptions are correct. Appropriately framed as documentation; not illusory novelty.

---

*Document applies the Scientific Analysis Discipline per AGENTS.md.*
