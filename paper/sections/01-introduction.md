# 1. Introduction

<!-- Funnel: TB burden → AI on CXR → centralized limits (privacy, heterogeneity, fairness)
     → FL as answer but FedAvg fragile + can be unfair → our contributions. -->

## Motivation
- TB disease burden; value of CXR triage; AI performance on CXR.
- Why centralized training is limiting: privacy, data silos, and distribution shift across
  sites.

## Problem
- Statistical heterogeneity (non-IID) degrades FedAvg.
- FL can amplify demographic unfairness (sex, age) — under-examined for TB.

## Contributions
1. A reproducible benchmark of heterogeneity-robust + fairness-aware FL for TB CXR on public
   datasets with per-image demographics (Shenzhen, Montgomery).
2. Empirical characterization of the **heterogeneity × fairness** interaction.
3. _(forward pointer)_ extension toward modality-incomplete FL (Objective 2 / future work).

## Paper structure
Brief roadmap of remaining sections.
