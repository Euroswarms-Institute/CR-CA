# CR-CA

**Version:** v1.5.0  
**Repository:** [IlumCI/CR-CA](https://github.com/IlumCI/CR-CA)  
**License:** Apache-2.0

CR-CA is a **causal reasoning and counterfactual analysis** stack for research and applications: structural causal models (SCMs), LLM-assisted workflows, deterministic simulation, counterfactuals, and specialized branches (quantitative trading, socioeconomic dynamics, corporate governance). The core library is published on PyPI as **`crca`**.

## AIXI research track

This repository also hosts an **AIXI** line of work—computable approximations to universal reinforcement learning for long-horizon research. **Theory, V0 status, roadmap, citations, and AIXI-specific install/usage are documented in [`aixi/README.md`](aixi/README.md)** (not duplicated here).

---

## Documentation

- **CR-CA user & API guides:** [`docs/`](docs/) (MkDocs; start at [`docs/index.md`](docs/index.md)).
- **AIXI track:** [`aixi/README.md`](aixi/README.md), [`aixi/IMPLEMENTATION_PLAN.md`](aixi/IMPLEMENTATION_PLAN.md).

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
pip install -e ".[aixi]"         # optional: pyaixi (VCS) + AIXI smoke deps
```

Using **uv** (recommended for locked local dev):

```bash
uv sync --extra dev --extra aixi
```

Optional heavy extras (e.g. `cvxpy`) are listed under `[project.optional-dependencies]` in [`pyproject.toml`](pyproject.toml).

---

## Developer workflow

| Task | Command |
|------|---------|
| Tests | `pytest` |
| AIXI smoke / Phase 0 | [`aixi/README.md`](aixi/README.md) (`make aixi-phase0`) |
| Format / lint | `black`, `ruff`, `mypy` (from `[dev]` extra) |

---

## Contributing

Pull requests should keep domain assumptions explicit in new code paths. **LaTeX- or theorem-heavy AIXI docs** should be reviewed with **Math & CS Wizard** before merge (see [`aixi/README.md`](aixi/README.md)).

---

## Changelog

- **v1.5.x** — AIXI v0 track alongside core CRC-A releases.
- Full history: [`docs/changelog/index.md`](docs/changelog/index.md).
