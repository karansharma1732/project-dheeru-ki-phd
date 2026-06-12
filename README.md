# Heterogeneity-Aware & Fairness-Aware Federated Learning for TB Detection

PhD research project on **federated learning (FL) for tuberculosis (TB) detection** from
chest X-rays and other clinical evidence, with two objectives:

1. **Objective 1 — Heterogeneity- + fairness-aware FL on chest X-rays (CXR).**
   The backbone pipeline: simulate non-IID clients in [Flower](https://flower.dev),
   benchmark heterogeneity-robust aggregation (FedAvg → FedProx → SCAFFOLD) and
   fairness-aware aggregation (FairFed, q-FFL), evaluated on public TB CXR datasets.
2. **Objective 2 — Modality-heterogeneous / missing-modality FL.**
   Different clients hold different TB evidence types (CXR + demographics, clinical
   tabular records, sputum-smear microscopy). The central novelty: the first
   TB-specific modality-incomplete federated benchmark.

> See [docs/implementation-plan.md](docs/implementation-plan.md) for the full feasibility
> analysis and build order.

## Repository layout

| Path | Purpose |
|------|---------|
| [docs/](docs/) | Research design: objectives, datasets, methodology, literature review, implementation plan |
| [paper/](paper/) | Manuscript: outline, abstract, and per-section drafts |
| [code/](code/) | Implementation (FL pipeline, models, experiments) |
| [data/](data/) | Dataset documentation and download/preparation notes (raw data is **not** committed) |
| [experiments/](experiments/) | Experiment configs, run logs, and results tracking |
| [references/](references/) | Bibliography (BibTeX) and per-paper reading notes |
| [notes/](notes/) | Meeting notes, ideas, and an open-questions log |

## Status

- [ ] Objective 1 — heterogeneity core (FedAvg / FedProx)
- [ ] Objective 1 — fairness module (FairFed / q-FFL)
- [ ] Objective 1 — paper draft
- [ ] Objective 2 — modality-incomplete FL benchmark
- [ ] TB Portals Data Use Agreement application

## Getting started

1. Read [docs/objective.md](docs/objective.md) and [docs/implementation-plan.md](docs/implementation-plan.md).
2. Review [docs/datasets.md](docs/datasets.md) and prepare data per [data/README.md](data/README.md).
3. Follow the phased build order in the implementation plan.
