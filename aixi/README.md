# AIXI research track (SWA-3)

Parent issue: board request to analyze listed AIXI-related sources and converge on a **computable, bleeding-edge implementation plan** (formulas, interfaces, and how pieces compose).

## Sources (one analysis file each)

| # | Source | Analysis file (Research Engineer) |
|---|--------|-----------------------------------|
| 1 | [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/56a225639da77e8f7c0409f6d5ba996b-Paper-Conference.pdf) | `analyses/01-neurips-2023.md` |
| 2 | [arXiv 2602.23242](https://arxiv.org/pdf/2602.23242) | `analyses/02-arxiv-2602-23242.md` |
| 3 | [pyaixi](https://github.com/sgkasselau/pyaixi) | `analyses/03-pyaixi-repo.md` |
| 4 | [arXiv 2502.15820](https://arxiv.org/html/2502.15820v2) | `analyses/04-arxiv-2502-15820.md` |
| 5 | [arXiv 2511.22226](https://arxiv.org/pdf/2511.22226) | `analyses/05-arxiv-2511-22226.md` |
| 6 | [arXiv 2505.21170](https://arxiv.org/html/2505.21170v2) | `analyses/06-arxiv-2505-21170.md` |

## Deliverables

- **Per paper:** structured notes in `analyses/` (problem, definitions, main theorems/algorithms, notation, what is implementable vs idealized, links to CRCA patterns if any).
- **Synthesis:** `IMPLEMENTATION_PLAN.md` is updated after all analyses land (CEO owns consolidation unless delegated).

## Repo context

Use existing `CR-CA` abstractions where they help (agents, templates, prediction hooks); do not force-fit causal machinery where it does not apply.
