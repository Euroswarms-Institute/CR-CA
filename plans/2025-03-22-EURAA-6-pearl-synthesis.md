# EURAA-6: Pearl Literature Summary

**Task:** Pearl literature summary (do-calculus, SCM, three-level hierarchy, identification)  
**Sources:** arxiv:1210.4852, arxiv:2512.12804, Pearl *Causality* (UCLA)  
**Author:** Research Engineer  
**Date:** 2025-03-22

---

## Part A: Synthesis (~500 words)

### Structural Causal Models (SCMs)

Pearl’s framework rests on structural causal models: systems of equations that assign each endogenous variable as a function of its direct causes and exogenous noise. Formally, \(X_i = f_i(\text{Pa}(X_i), U_i)\) where \(\text{Pa}(X_i)\) are the parents in a causal graph and \(U_i\) are independent noise variables. The graph encodes which variables influence which; the equations specify the functional form. This setup supports both probabilistic prediction (observational) and causal reasoning (interventions and counterfactuals). SCMs unify graphical, structural-equation, and potential-outcome frameworks.

### Three-Level Causal Hierarchy

Pearl’s “ladder of causation” distinguishes three levels:

- **Level 1 — Association:** “Is X associated with Y?” Answered by conditional probabilities, correlation, regression. No causal semantics; rooster’s crow and sunrise covary but causation is unspecified. Statistical tools of the early 20th century operate here.

- **Level 2 — Intervention:** “If I do X, how will Y change?” Requires causal knowledge beyond association. The \(do(X=x)\) operator models setting X by intervention, breaking its usual causes. \(P(Y\mid do(X)) \neq P(Y\mid X)\) when confounding exists. RCTs approximate \(do\)-experiments.

- **Level 3 — Counterfactual:** “Would Y have been different had X been x, given what actually happened?” Requires a full SCM: we abduce the factual noise from the observation, then apply the counterfactual intervention and predict. Single-event causation (“Was it the exposure that caused this outcome?”) lives at this level.

Questions at level \(i\) can only be answered with information from level \(j \geq i\).

### Do-Calculus and Identification

The do-calculus (Pearl, *Causality* Ch. 3.4) gives inference rules to rewrite \(P(Y\mid do(X))\) into observational quantities when possible. Completeness was established by Huang & Valtorta (2006) and Shpitser & Pearl (2006). Three rules allow exchanging \(do\)-terms with observe-terms under graph conditions. Graphical criteria (backdoor, frontdoor, Tian–Shpitser) characterize identifiability. Applications extend to mediation, transportability, and meta-synthesis. When identification fails, only bounds or additional assumptions can help.

### Identification Strategies

- **Backdoor:** Adjust for a set Z that blocks all backdoor paths from treatment to outcome. \(P(Y\mid do(X)) = \sum_z P(Y\mid X,z)\,P(z)\).

- **Frontdoor:** Use a mediator M that (1) fully mediates X→Y, (2) has no unmeasured confounding with X, (3) has all backdoor paths to Y blocked by X. Leads to a two-step formula.

- **Instrumental variable:** Z affects X, affects Y only through X, and is independent of unmeasured causes of Y. Under linearity, \(\beta = \text{Cov}(Z,Y)/\text{Cov}(Z,X)\).

- **ID algorithm:** General recursion over graph components; returns a formula when identifiable, or declares non-identifiability when latent confounding blocks identification.

### Recent Extensions (arxiv:2512.12804)

Beckers (2025) proposes a semantics for counterfactual probabilities that generalizes Pearl’s: it applies to probabilistic causal models that cannot be extended to full SCMs. He navigates the Pearl–Dawid debate: agreeing with Dawid on rejecting universal determinism and unrealistic variables, but with Pearl that a general counterfactual semantics is possible. The Markov condition and causal completeness play central roles. This suggests the Pearlian framework remains extensible under weaker assumptions.

---

## Part B: Scientific Analysis of This Synthesis

### 200-Word Summary

This synthesis summarizes Pearl’s causal framework: SCMs, the three-level hierarchy (Association → Intervention → Counterfactual), do-calculus, and identification. It draws on arxiv:1210.4852, arxiv:2512.12804, and Pearl’s *Causality*. The synthesis is a **literature review**, not original research. It compresses well-established theory into ~500 words for use in downstream CRCA comparison and math-audit tasks.

### Scientific Validity

- **~85%:** The synthesis accurately reflects Pearl’s framework. Claims about backdoor, frontdoor, do-calculus rules, and the hierarchy match canonical sources. Minor risks: oversimplification of the ID algorithm; the Beckers summary is based on abstract only.
- **~10%:** Some nuance may be lost (e.g., Markov equivalence, faithfulness, completeness proofs).
- **~5%:** Possible conflation between textbook presentation and cutting-edge extensions.

### Scientific Novelty

- **~0%:** No novel claims. This is synthesis of existing literature.
- **~95%:** Correctly presented as a review/summary.

### Classification: Science vs. Art vs. Other

- **Science (as literature review):** ~70% — Systematic summarization of a scientific framework.
- **Engineering/documentation:** ~25% — Prepared for a software project (CRCA).
- **Art:** ~5% — Minimal; some phrasing choices, but no creative claims.

### Math Correctness

- **~90%:** The stated formulas (backdoor, frontdoor, IV) are correct. The description of the three levels is standard.
- **~10%:** Risk of typo or omitted condition (e.g., positivity for backdoor). No formal proof-check performed.

### Novel vs. Illusion-of-Novel Research

- **~95%:** This deliverable does *not* claim novelty. It explicitly reviews prior work.
- **~5%:** A reader could misread the Beckers summary as original contribution; the document labels it “Recent Extensions” and cites the source.

### Hypothesis Probabilities (Summary)

| Hypothesis | P |
|------------|---|
| Synthesis is factually accurate for Pearl’s framework | 0.85 |
| Synthesis omits important nuance | 0.10 |
| Math/formulas are correct | 0.90 |
| Deliverable is appropriately modest (no false novelty) | 0.95 |
| Beckers summary may need revision after full-paper read | 0.30 |

---

## Email to Author (Synthesis Author = Research Engineer)

**To:** Research Engineer (self)  
**Re:** Scientific analysis of EURAA-6 Pearl synthesis

The synthesis you produced for EURAA-6 is a **competent literature review** of Pearl’s causal framework. It correctly captures SCMs, the three-level hierarchy, do-calculus, and main identification strategies. The math (backdoor, frontdoor, IV) is accurate to the best of our verification.

**Strengths:** Clear structure, appropriate scope, explicit sourcing. No false claims of novelty. Suitable as input for CRCA comparison (EURAA-7) and math audit (EURAA-8).

**Caveats:** (1) The Beckers (2512.12804) summary is abstract-only; full-paper read may warrant updates. (2) The ID-algorithm description is high-level; the CRCA implementation uses a conservative, g-formula-only variant. (3) Consider adding explicit mention of d-separation and the three do-calculus rules for completeness.

**Verdict:** Scientifically valid as a summary. Not novel. Correctly classified as science-adjacent (literature review). Math appears correct. Not an instance of illusory novelty—appropriately framed as prior-art synthesis.

---

*Document applies the Scientific Analysis Discipline per AGENTS.md.*
