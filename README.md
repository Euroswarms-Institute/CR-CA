# CR-CA / AIXI

**Version:** v1.5.0  
**Repository:** [IlumCI/CR-CA](https://github.com/IlumCI/CR-CA)  
**License:** Apache-2.0

CR-CA is a **causal reasoning and counterfactual analysis** stack for research and applications: structural causal models (SCMs), LLM-assisted workflows, and specialized branches (quantitative trading, socioeconomic dynamics, corporate governance). In parallel, this repository hosts an **AIXI research track**—computable approximations to universal reinforcement learning—aimed at long-horizon **AGI research**, not a claim of full Solomonoff optimality.

---

## AIXI: what the theory targets

AIXI is a **Bayes-optimal reinforcement learner** under a mixture prior over environments. Informally, at history \(h_{<t}\) the agent maintains a posterior over models \(\nu\) and acts to maximize expected **discounted return** with discount \(\gamma \in (0,1)\). Writing \(\xi(\cdot \mid h_{<t})\) for the **Bayes mixture** over a countable or finite class \(\mathcal{M}\) of environment hypotheses,

$$
V_\xi^\pi(h_{<t}) = \mathbb{E}_{\xi,\pi}\Big[\sum_{k=t}^{\infty} \gamma^{k-t} r_k \,\Big|\, h_{<t}\Big], \qquad
\pi_\xi^\ast \in \arg\max_\pi V_\xi^\pi(h_{<t}).
$$

The **AIXI policy** is \(\pi_\xi^\ast\) when \(\xi\) is a **universal** semimeasure over computable environments; that construction is **not computable** in finite time. This codebase implements **finite** \(\mathcal{M}\), **budgeted** planning, and explicit **revert/replay** contracts so that \(\xi\) updates remain ordinary Turing-bounded operations—see [`aixi/IMPLEMENTATION_PLAN.md`](aixi/IMPLEMENTATION_PLAN.md) and [`aixi/SUPERTASK_BOUNDARY.md`](aixi/SUPERTASK_BOUNDARY.md).

---

## V0 implementation reality (what exists today)

| Area | Status | Where |
|------|--------|--------|
| **Phase 0 parity** | Symbol-level agreement between `pyaixi` CTW rollouts and a manual CTW driver on `CoinFlip` | [`aixi/aixi/parity/`](aixi/aixi/parity/), [`aixi/experiments/PHASE0_PYAIXI_PARITY_REPORT.md`](aixi/experiments/PHASE0_PYAIXI_PARITY_REPORT.md) |
| **Mixture \(\xi\) adapter** | `PyAixiCTWBitMixture` bridging CTW + search | [`aixi/aixi/models/ctw_pyaixi.py`](aixi/aixi/models/ctw_pyaixi.py) |
| **Planning (Family A)** | Root UCT-style action selection with revert checks on \(\xi\) | [`aixi/aixi/planning/mcts.py`](aixi/aixi/planning/mcts.py), smoke in [`aixi/experiments/run_smoke.py`](aixi/experiments/run_smoke.py) |
| **Self-AIXI (Family B)** | Policy mixture + \(Q_{\zeta\xi}\)-style scaffolding | [`aixi/aixi/planning/self_aixi/`](aixi/aixi/planning/self_aixi/) |
| **AIQI (Family C)** | Return-mixture / augmentation skeleton | [`aixi/aixi/planning/aiqi/`](aixi/aixi/planning/aiqi/) |
| **Formal CI properties** | Cross-family tests (revert stack, \(\omega\) normalization, augmentation schedule, budgets) | [`aixi/formal_ci.py`](aixi/formal_ci.py), `tests/test_aixi_*.py` |
| **Optional signals** | Intrinsic / empowerment-style hooks (gated) | [`aixi/aixi/signals/`](aixi/aixi/signals/) |
| **Quantum toy** | Classical CPTP toy env for lab experiments | [`aixi/experiments/quantum_toy/`](aixi/experiments/quantum_toy/) |

**Smoke entrypoint (recommended):**

```bash
uv run --extra aixi python -m aixi.experiments.run_smoke
# or
make aixi-phase0
```

Source analyses and paper-to-module mapping live under [`aixi/analyses/`](aixi/analyses/) and [`aixi/modules/`](aixi/modules/); the consolidated design doc is [`aixi/IMPLEMENTATION_PLAN.md`](aixi/IMPLEMENTATION_PLAN.md).

---

## Roadmap: from V0 toward broader AGI research

The implementation plan phases (abbreviated) are:

1. **Phase 0 — Parity:** fixed `pyaixi` config on a toy env (done; see report above).
2. **Phase 1 — Unified \(\xi\) API:** one `MixtureEnvModel` for MCTS and Self-AIXI heads (in progress; see tests and `mixture` modules).
3. **Phase 2 — Self-AIXI v0:** finite policy class, tabular \(Q\) on small envs; compare to MC-AIXI baseline.
4. **Phase 3+ — Scale & AIQI:** function approximation; AIQI-style return mixtures with explicit **on-policy** training assumptions.
5. **Later:** optional empowerment / FEP regularizers, joint prediction interfaces, quantum **classical** simulators—each gated as research, not core product claims.

This roadmap is **not** a promise of timelines; it is an ordering that keeps **regression baselines** and **computable** interfaces ahead of speculative extensions.

---

## CR-CA framework (causal stack)

The broader product is a modular Python package (`crca` on PyPI) with:

- **Core agent** (`CRCA.py`, `crca_core/`) — extraction, simulation, counterfactuals.
- **Branches** — `crca_sd`, `crca_cg`, `CRCA-Q`-style trading workflows.
- **Infrastructure** — `utils/`, `tools/`, `templates/`, `schemas/`, MCP hooks.

**Extended documentation** (installation, features, API-style guides) is maintained in the MkDocs site under [`docs/`](docs/) (see [`docs/index.md`](docs/index.md) and [`docs/getting-started/installation.md`](docs/getting-started/installation.md)).

---

## Installation

**Requirements:** Python **≥ 3.10**.

```bash
# PyPI
pip install crca

# From source (editable)
git clone https://github.com/IlumCI/CR-CA.git
cd CR-CA
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[aixi]"         # adds pyaixi (VCS) + AIXI smoke deps
```

Using **uv** (recommended for locked local dev):

```bash
uv sync --extra dev --extra aixi
```

Optional heavy extras (e.g. `cvxpy` for optimization) are listed under `[project.optional-dependencies]` in [`pyproject.toml`](pyproject.toml).

---

## Developer workflow

| Task | Command |
|------|---------|
| AIXI Phase 0 smoke | `make aixi-phase0` or `uv run --extra aixi python -m aixi.experiments.run_smoke --help` |
| Tests (default quiet) | `pytest` |
| AIXI-related markers | `pytest -m "aixi_p1 or aixi_p2"` (see `pyproject.toml` `[tool.pytest.ini_options]` markers) |
| Format / lint | `black`, `ruff`, `mypy` (from `[dev]` extra) |

---

## Contributing & review

Pull requests should keep **finite-model** and **budget** assumptions explicit in new code paths. For LaTeX-heavy README or plan changes, pair with **Math & CS Wizard** for notation consistency before merge.

---

## References (primary)

1. M. Hutter, *Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability*, Springer, 2005. ([AIXI definition & Bayes mixture](https://www.hutter1.net/ai/uaibook.htm))
2. J. Veness et al., *A Monte-Carlo AIXI Approximation*, JAIR 2011 (MC-AIXI / CTW line; implementation lineage via `pyaixi`).
3. J. Veness et al., *Practical Monte Carlo AIXI with Context Tree Weighting*, and related CTW literature referenced in `pyaixi`.
4. E. Catt et al., *Self-Predictive Universal AI*, NeurIPS 2023 ([PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/56a225639da77e8f7c0409f6d5ba996b-Paper-Conference.pdf)) — Self-AIXI / \(Q_{\zeta\xi}\) line; see [`aixi/analyses/01-neurips-2023.md`](aixi/analyses/01-neurips-2023.md).
5. M. Hutter & coauthors on **AIQI** / return-induction (see [`aixi/analyses/02-arxiv-2602-23242.md`](aixi/analyses/02-arxiv-2602-23242.md) for this repo’s working notes).

Older CR-CA changelog and feature tables previously embedded in this file are superseded by **versioned docs** in [`docs/changelog/`](docs/changelog/index.md) and the structured guides under [`docs/`](docs/).

---

## Changelog (high level)

- **v1.5.x** — AIXI v0 track: parity, MCTS integration, formal CI harness, implementation plan synthesis.
- Prior CR-CA feature history: see [`docs/changelog/index.md`](docs/changelog/index.md).
