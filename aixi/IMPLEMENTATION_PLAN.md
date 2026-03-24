# AIXI — implementation plan (living document)

**Status:** scaffold — awaiting six source analyses in `analyses/`.

## 0. Goal

Ship a **computable** AIXI-class agent design that:

- States explicit approximations (finite horizon, model class, search budget, environment API).
- Maps each symbol in core equations to code artifacts (modules, data structures, interfaces).
- Calls out dependencies between papers/repos and resolves notation clashes.

## 1. Notation ledger

*(Fill after analyses: shared symbols, discounts, priors, observation/action spaces.)*

## 2. Environment interface

*(API: step, obs encoding, reward, episodic vs continuing, stochasticity assumptions.)*

## 3. Model class & prior

*(Bayes / SOLMS / neural approximators — tie to papers.)*

## 4. Planning / search

*(Expectimax depth, sampling, MCTS, MC-AIXI, etc.)*

## 5. Implementation map → repository layout

*(Proposed package names, core loops, tests, compute budget knobs.)*

## 6. Phased rollout

1. Minimal tick loop + toy env  
2. Plug in first approximate model  
3. Scale search / validate against `pyaixi` or published baselines where applicable  

## 7. Open risks

*(Theoretical gaps, non-computability limits, where we knowingly bound the problem.)*

---

### Analysis checklist

- [ ] `analyses/01-neurips-2023.md`
- [ ] `analyses/02-arxiv-2602-23242.md`
- [ ] `analyses/03-pyaixi-repo.md`
- [ ] `analyses/04-arxiv-2502-15820.md`
- [ ] `analyses/05-arxiv-2511-22226.md`
- [ ] `analyses/06-arxiv-2505-21170.md`
