# Paper 1 — Outline

**Working title:** Heterogeneity-Aware and Fairness-Aware Federated Learning for
Tuberculosis Detection on Chest X-rays

**Target venue:** TBD (medical-imaging / ML-for-health)
**Status:** outline

## Section drafts (in [sections/](sections/))
1. [Abstract](sections/00-abstract.md)
2. [Introduction](sections/01-introduction.md)
3. [Related Work](sections/02-related-work.md)
4. [Methods](sections/03-methods.md)
5. [Experimental Setup](sections/04-experiments.md)
6. [Results](sections/05-results.md)
7. [Discussion](sections/06-discussion.md)
8. [Conclusion](sections/07-conclusion.md)

## One-line argument
Standard federated TB detection degrades under real-world heterogeneity and can be unfair
across demographic subgroups; combining a heterogeneity-robust aggregator (FedProx) with a
fairness-aware aggregator (FairFed) recovers accuracy *and* closes subgroup gaps, with the
fairness benefit growing as heterogeneity worsens.

## Key claims to support with experiments
- C1: non-IID degrades FedAvg vs centralized/local (RQ1.1).
- C2: FedProx/SCAFFOLD recover accuracy + convergence stability (RQ1.2).
- C3: FairFed/q-FFL reduce subgroup gaps at minimal accuracy cost; benefit ↑ with α↓ (RQ1.3).

## Figures/tables (planned — see [figures/](figures/))
- Fig 1: system overview (clients, partitions, aggregation, fairness module).
- Fig 2: accuracy vs α for FedAvg / FedProx / SCAFFOLD.
- Fig 3: subgroup AUC gaps with/without fairness module.
- Table 1: dataset/partition summary.
- Table 2: main results (accuracy, sens, spec, AUC, F1 + fairness metrics + comms cost).
