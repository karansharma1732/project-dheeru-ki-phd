# Research Objectives

## Title (working)
Heterogeneity-Aware and Fairness-Aware Federated Learning for Tuberculosis Detection
Across Heterogeneous Evidence Types

## Problem statement
Tuberculosis remains a leading cause of death from a single infectious agent. AI models
for TB detection from chest X-rays (CXR) are typically trained centrally, which raises
privacy concerns and, more importantly, fails under the *heterogeneity* of real-world
deployment: data is spread across institutions with different scanners, populations, TB
prevalence, and even different *modalities* of evidence (CXR, clinical records, sputum
microscopy). Federated learning (FL) keeps data local, but standard FL (FedAvg) degrades
under statistical heterogeneity and can encode demographic unfairness.

## Objective 1 — Heterogeneity- and fairness-aware FL on CXR
Build an FL pipeline that is robust to non-IID data **and** fair across demographic
subgroups (sex, age band) for TB detection on chest X-rays.

- **RQ1.1** How much does statistical heterogeneity (label skew, source/domain shift)
  degrade federated TB detection relative to centralized and local-only baselines?
- **RQ1.2** Do heterogeneity-robust aggregators (FedProx, SCAFFOLD) recover accuracy and
  convergence stability under severe non-IID partitions?
- **RQ1.3** Can fairness-aware aggregation (FairFed, q-FFL) reduce subgroup performance
  gaps without sacrificing overall accuracy, and does the benefit grow with heterogeneity?

## Objective 2 — Modality-heterogeneous / missing-modality FL
Generalize the framework to clients holding **different modalities** of TB evidence, where
no single client (and no patient) has all modalities.

- **RQ2.1** How can FL aggregate knowledge across clients that hold disjoint modalities
  (CXR + demographics, clinical tabular, sputum-smear microscopy)?
- **RQ2.2** Do modality-incomplete FL methods (ClusMFL, feature-imputation networks)
  outperform modality-dropout and single-modality baselines?
- **RQ2.3** How does performance scale with the *modality-availability ratio* across
  clients?

> **Reframing note.** The originally proposed per-patient fusion of CXR + clinical +
> sputum is infeasible on public data (no public dataset pairs all three for the same
> patients). Objective 2 is therefore framed as modality-*heterogeneous* FL across
> clients, which is both feasible and novel. See
> [implementation-plan.md](implementation-plan.md) §9.

## Expected contributions
1. A reproducible benchmark of heterogeneity-robust + fairness-aware FL for TB CXR.
2. Empirical evidence on the interaction between heterogeneity and demographic fairness.
3. The first TB-specific modality-incomplete federated learning benchmark.

## Scope and non-goals
- **In scope:** simulation-mode FL (Flower), public TB datasets, fairness on Shenzhen +
  Montgomery (the only sets with per-image demographics).
- **Out of scope / stretch:** differential privacy (Opacus), foundation-model fusion
  (FedPIA), and real multi-site deployment — treated as optional extensions.
