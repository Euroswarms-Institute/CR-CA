# EURAA-5 Research Plan: Judea Pearl × CRCA.py

**Goal:** Research & Development of Causal & Counterfactual reasoning systems in LLMs/LRMs  
**Created:** 2025-03-22  
**Owner:** Research Director (CEO)  
**Executor:** Research Engineer

---

## 1. Objectives

1. **Pearl literature review** — Research Judea Pearl's causal/counterfactual framework (do-calculus, SCMs, three-level hierarchy) using arxiv and canonical papers.
2. **CRCA.py comparison** — Map the architecture in `CRCA.py` to Pearl's framework: what aligns, what diverges, where the math and semantics differ.
3. **Scientific analysis discipline** — Apply the attached scientific-analysis prompt to every research artifact and task; maintain hypotheses and probability estimates throughout.

---

## 2. Key References (Pearl / arxiv)

- **Do-calculus:** [arxiv:1210.4852](https://arxiv.org/abs/1210.4852) — "The Do-Calculus Revisited"
- **Counterfactuals:** [arxiv:2512.12804](https://arxiv.org/abs/2512.12804) — "Causal Counterfactuals Reconsidered"
- **Pearl SCM / hierarchy:** *Causality* (Pearl 2009), UCLA stat_ser (r402, r481, r475-cacm)
- **Three levels:** Association → Intervention → Counterfactual

---

## 3. CRCA.py Architecture (High-Level)

- **Causal graph:** `causal_graph` dict + `rustworkx` `PyDiGraph`; edges store strength, confidence, relation type
- **Interventions:** `CounterfactualScenario` with `interventions`, `expected_outcomes`; `do(x)`-like behavior via graph surgery
- **SCM simulation:** Nonlinear SCM (tanh/identity), standardization stats, `_AdaptiveInterventionSampler`, `_GraphUncertaintySampler`
- **Tools:** `extract_causal_variables`, `generate_causal_analysis` (LLM-driven); deterministic simulation via `predict`/counterfactual logic
- **Excel bridge:** Optional SCM bridge, dependency graph, evaluation engine

---

## 4. Board Directive (2025-03-22)

> Have the Research Engineer fix up the CR-CA repo to stay true to the Causal & Counterfactual reasoning of Judea Pearl. Don't make it non-deterministic completely, but have all the important process not just be prompt tricks — **actual causal & counterfactual reasoning**.

**Implication:** Research subtasks (6–8) inform *what* to fix; implementation subtask (9) does the fix.

---

## 5. Research Engineer Tasks (Subtasks)

| # | Task | Description |
|---|------|-------------|
| 1 | Pearl literature summary | Read arxiv papers, write ~500-word synthesis: do-calculus, SCM, three-level hierarchy, identification |
| 2 | CRCA ↔ Pearl mapping | Create mapping doc: which CRCA constructs map to Pearl (graph, intervention, counterfactual); where they diverge |
| 3 | Math audit | Check CRCA math (linear/nonlinear SCM, intervention propagation) against Pearl semantics |
| 4 | **Implement Pearl-aligned reasoning** | Fix CR-CA repo: replace prompt tricks with actual causal/counterfactual machinery; keep deterministic SCM core |
| 5 | Scientific analysis | Apply the scientific-analysis prompt to CRCA.py and each deliverable; maintain hypothesis probabilities |

---

## 6. Scientific Analysis Prompt (Apply to All Work)

The Research Engineer MUST ask itself this for every task and artifact:

> Please take a look at the attached project and provide a careful critical analysis of it from a scientific perspective. Start with a 200-word summary of the project.
>
> Focus on answering the following questions:
>
> - To what extent is this project scientifically valid?
> - To what extent is this project scientifically novel?
> - Would you classify this as science, art, or something else, and why?
> - Is the math correct throughout or are there errors?
>
> There are many interesting and novel research projects going on using LLMs. There are also people who have been fooled into believing they're doing interesting and novel research when they aren't. To what extent is this one or the other of those?
>
> Please conclude with an email to the author, summarizing your analysis of the project. Think about this analysis as hard as you can. Double- and triple-check your conclusions. Maintain multiple hypotheses about the project simultaneously throughout your analysis, and at each step assign a probability estimate to the truth of each hypothesis. Thanks!

---

## 7. Subtask Creation (Paperclip)

- Create subtasks under EURAA-5 with `parentId: 90cd81ba-74a7-4cf2-be2c-e2a2ff06d7a8`, `goalId: b058e356-eabd-439e-a7a5-84a90971bac3`
- Assign to Research Engineer (`806b117a-c227-40bd-b03a-c1d0f7b3a4de`)
