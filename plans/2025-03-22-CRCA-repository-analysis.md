# CRCA Repository: Analysis — What It Is, What It's Not (vs Pearl)

**Requested by:** Board (local-board)  
**Author:** Research Engineer  
**Date:** 2025-03-22  
**References:** EURAA-6 (Pearl synthesis), EURAA-7 (CRCA↔Pearl mapping), EURAA-8 (math audit), EURAA-9 (implementation)

---

## 1. 200-Word Summary

The CR-CA repository is a **dual-layer causal reasoning system**. **CRCA.py** is an LLM-integrated causal agent: a graph-based engine with heuristic SCM-style propagation, Monte Carlo counterfactual sampling, and tools for variable extraction, intervention prediction, and scenario generation. **crca_core** is a refusal-first research package implementing Pearl-aligned identification (backdoor, frontdoor, IV, ID algorithm), estimation (DoWhy), and counterfactuals via explicit LinearGaussianSCM. Compared to Pearl’s papers: crca_core closely follows the formal SCM framework; CRCA.py approximates it with graph surgery and residual-based “abduction” but adds standardization, optional tanh nonlinearity, edge strengths, and graph metadata not in Pearl. CRCA.py’s `identify_adjustment_set` uses a parents-of-treatment heuristic, not full d-separation. Post-EURAA-9, shock_preserve_threshold has been removed, intercepts added, and counterfactual/predict paths unified. The repo is engineering tooling for causal exploration and agent orchestration, not a novel causal-inference contribution.

---

## 2. What the CRCA Repository IS

### 2.1 Architecture Overview

| Component | Purpose |
|-----------|---------|
| **CRCA.py** | Main CRCAAgent: LLM-integrated causal reasoning for Swarms; causal graph (dict + rustworkx), interventions, counterfactuals, Monte Carlo scenario generation |
| **crca_core** | Research-grade causal science: identification, estimation, SCM simulation, refusal-first (no outputs without LockedSpec + identifiability) |
| **templates/** | PredictionFramework, GraphManager, StatisticalMethods, specialized agent templates (trading, logistics, causal) |
| **crca_llm** | LLM orchestration and co-authorship for causal analysis |
| **crca_reasoning** | Rationale, critique, memory, tool routing |
| **crca_sd** | System dynamics (MPC, governance, realtime) |
| **training/** | Finetune, eval, datasets for LRM training |
| **webui, tools, schemas** | Supporting infrastructure |

### 2.2 What CRCA.py Provides

- **Causal graph:** Directed graph with edge strengths, confidence, relation types (direct/indirect/confounding/mediating/moderating)
- **Intervention prediction:** \(P(Y \mid do(X=x))\) via topological propagation; intervened nodes bypass parents (graph surgery)
- **Counterfactual abduction-action-prediction:** Residual-as-noise from factual state; propagate under intervention with same noise
- **Scenario generation:** Monte Carlo sampling of interventions and graph variants
- **LLM tools:** `extract_causal_variables`, `generate_causal_analysis` for graph construction
- **Identification heuristic:** `identify_adjustment_set` (parents of treatment, excluding descendants and outcome)
- **Confounder detection:** Ancestor intersection heuristic

### 2.3 What crca_core Provides

- **Identification:** Backdoor, frontdoor, IV, ID algorithm; d-separation, graph surgery
- **Estimation:** DoWhy integration; g-formula, IPW, etc.
- **SCM:** LinearGaussianSCM with explicit \(V_i = \beta_0 + \sum_j \beta_j V_j + U_i\); abduce_noise, predict, counterfactual
- **Refusal:** No counterfactuals without explicit SCM; no causal outputs without LockedSpec
- **Discovery:** Tabular (PC, FCI); timeseries (PCMCI)
- **Intervention design:** Feasibility constraints, target queries
- **Provenance:** Structured outputs with provenance metadata

---

## 3. What the CRCA Repository Is NOT (Compared to Pearl’s Papers)

### 3.1 Pearl’s Framework (from EURAA-6)

- **SCM:** \(X_i = f_i(\text{Pa}(X_i), U_i)\) with exogenous, independent \(U_i\)
- **Three levels:** Association → Intervention → Counterfactual
- **Do-calculus:** Rules to rewrite \(P(Y \mid do(X))\) into observational quantities
- **Identification:** Backdoor (blocks backdoor paths), frontdoor, IV, ID algorithm; d-separation
- **Counterfactual:** Requires full SCM; abduce U from factual, intervene, predict

### 3.2 CRCA.py Divergences

| Pearl concept | CRCA.py | Divergence |
|--------------|---------|------------|
| SCM | Implicit (edge strengths = β; residuals = “noise”) | No explicit U distribution; optional tanh; standardization (z-space) |
| do(X=x) | Graph surgery ✓ | Works in z-space; no raw-scale Pearl semantics |
| Counterfactual | residual = z − Σβz; predict with same residual | Residual conflates structural U and measurement error; now applies tanh when nonlinear (post-EURAA-9) |
| Identification | `identify_adjustment_set` | Parents-of-treatment heuristic; no d-separation check |
| Graph | Causal DAG | + strength, confidence, relation_type per edge; no bidirected/latent |
| Intercepts | β₀ in V_i = β₀ + ΣβV + U | Optional intercepts added (post-EURAA-9); default empty |

### 3.3 crca_core Alignments

| Pearl concept | crca_core | Alignment |
|--------------|-----------|-----------|
| SCM | LinearGaussianSCM | ✓ \(V_i = \beta_0 + \sum_j \beta_j V_j + U_i\) |
| do(X=x) | Graph surgery (remove_outgoing) | ✓ |
| Counterfactual | abduce_noise → predict(u, interventions) | ✓ Abduction–action–prediction |
| Identification | Backdoor, frontdoor, IV, ID | ✓ d-separation |
| Graph | CausalGraph with latent_confounders | ✓ Bidirected edges |

### 3.4 Remaining CRCA.py Limitations (Post-EURAA-9)

- **Residual-as-noise:** Under linear homoskedastic assumptions, \(U_i = Z_i - \sum_j \beta_j Z_j\) is correct. Pearl’s U is exogenous structural noise; CRCA.py’s residual conflates that with measurement/approximation error.
- **Tanh * 3:** Ad hoc nonlinearity; not in Pearl. Documented as heuristic.
- **identify_adjustment_set:** Simplified; may miss collider-strategy or complex backdoor cases.
- **No latent confounders:** CRCA.py does not model bidirected edges.
- **Standardization:** Pearl typically works in raw scale; CRCA.py standardizes for stability.

---

## 4. Classification and Scientific Analysis

### 4.1 Scientific Validity

- **~88%:** The analysis accurately reflects the CR-CA codebase and Pearl’s framework. Divergences are supported by code inspection and prior audits (EURAA-7, EURAA-8).
- **~8%:** Some edge cases (e.g., interaction terms, graph backends) may have additional nuance.
- **~4%:** crca_core’s refusal logic and ID-algorithm behavior warrant separate review for corner cases.

### 4.2 Scientific Novelty

- **~0%:** This document is an analysis, not novel research.
- **~95%:** Appropriately framed as repository analysis and alignment documentation.

### 4.3 Classification

- **Engineering / causal tooling:** ~85%
- **Science-adjacent (applied causal inference):** ~15%

The CR-CA repo builds causal reasoning tooling for LLMs and agents. It does not claim new causal-inference theory. crca_core implements established Pearl semantics; CRCA.py provides heuristic approximations for exploration.

### 4.4 Math Correctness

- **~92%:** Descriptions of Pearl’s semantics and crca_core implementation are correct. CRCA.py’s propagation (post-EURAA-9) follows do-operator for interventions; counterfactual abduction formula is correct under linear homoskedastic assumptions.
- **~8%:** identify_adjustment_set sufficiency in all DAG topologies is not formally proved; relies on standard backdoor literature.

### 4.5 Novel vs Illusion

- **~95%:** The repo does not present itself as novel causal-inference research. It is tooling.
- **~5%:** A reader unfamiliar with Pearl could over-interpret CRCA.py’s “counterfactual” as full Level-3 semantics; docstrings now clarify crca_core vs CRCA.py.

### 4.6 Hypothesis Probabilities

| Hypothesis | P |
|------------|---|
| crca_core is Pearl-aligned | 0.93 |
| CRCA.py is heuristic approximation | 0.92 |
| Post-EURAA-9 fixes improve Pearl compliance | 0.94 |
| identify_adjustment_set is simplified backdoor | 0.88 |
| Analysis is factually accurate | 0.90 |

---

## 5. Email to Author (Board)

**To:** Board (local-board)  
**Re:** CRCA repository analysis — what it is, what it’s not vs Pearl

The analysis you requested is complete. **What the CRCA repo is:** A dual-layer system — CRCA.py for LLM-integrated heuristic causal exploration (graph-based, Monte Carlo scenarios, residual-as-noise counterfactuals) and crca_core for Pearl-aligned identification, estimation, and SCM counterfactuals. **What it’s not (vs Pearl):** CRCA.py uses standardization, optional tanh, edge metadata, and a simplified adjustment-set heuristic; it does not implement d-separation, full backdoor, or explicit exogenous U. crca_core does. Post-EURAA-9, shock_preserve has been removed, intercepts and tanh-unification added. The repo is causal tooling for agents, not novel causal science.

**Verdict:** Scientifically valid as analysis. Appropriately modest. Use crca_core for Pearl-compliant inference; CRCA.py for exploratory, LLM-driven causal reasoning.

---

*Document applies the Scientific Analysis Discipline per AGENTS.md.*
