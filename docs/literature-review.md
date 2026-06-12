# Literature Review

Living document. Group papers by theme; keep one-line takeaways and a "relevance to us"
note. Full BibTeX in [../references/references.bib](../references/references.bib);
detailed per-paper notes in [../references/notes/](../references/notes/).

## A. Federated learning — aggregation under heterogeneity
- **FedAvg** — McMahan et al., AISTATS 2017. Baseline weighted averaging.
- **FedProx** — Li et al., MLSys 2020 (arXiv 1812.06127). Proximal term; +22% avg accuracy
  vs FedAvg in highly heterogeneous settings. → our primary robustness method.
- **SCAFFOLD** — Karimireddy et al., ICML 2020. Control variates; struggles under partial
  participation (NIID-Bench).
- **NIID-Bench** — Li et al., ICDE 2022 (arXiv 2102.02079). Standard non-IID partitioning &
  evaluation. → reuse partitioning code.
- **Non-IID quantification** — Hsu et al. 2019 (arXiv 1909.06335). Dirichlet label skew.

## B. Fairness in FL
- **q-FFL / q-FedAvg** — Li et al., ICLR 2020 (arXiv 1905.10497). Performance fairness.
- **FairFed** — Ezzeldin et al., AAAI 2023 (arXiv 2110.00857). Group-fairness aggregation.
  → primary fairness method.
- **Agnostic FL (AFL)** — min-max worst-client. Lower priority.

## C. Modality-heterogeneous / missing-modality FL
- **ClusMFL** — Wang et al. 2025 (arXiv 2502.12180). Feature clustering + contrastive
  alignment + modality-aware aggregation. → strongest Objective 2 template.
- **Feature Imputation Network** — Poudel et al., MIUA 2025 (arXiv 2505.20232). Lightweight
  bottleneck-feature reconstruction. → Colab-friendly.
- **FedPIA** — Saha et al., AAAI 2025 (arXiv 2412.14424). Adapter permutation via Wasserstein
  barycenters; heavy. → optional/stretch.

## D. Multimodal fusion (CXR + clinical)
- **DeepCOVID-Fuse** — Ye et al., Bioengineering 2023 (arXiv 2301.08798). CXR+clinical
  AUC 0.842 vs 0.807 (CXR-only). → motivating evidence for fusion; not TB.
- **MDF-Net**, **RSNA transformer-fusion** — additional fusion precedents.

## E. TB-specific AI / FL
- **TBX11K** — Liu et al., CVPR 2020. Largest TB CXR set with bbox.
- **CODA TB DREAM** — cough+clinical; clinical-only AUROC 0.817 (PMC12502651). Motivating
  evidence for tabular TB signal; not CXR.
- **HPFL** (hyper-network personalized FL for TB), **FedARC** (adaptive contrastive FL for
  multi-center TB CXR). → existing TB FL precedents; **differentiate**: neither addresses
  *fairness* (our Obj. 1) or *modality-incompleteness* (our Obj. 2). Confirm methods from
  primary venues (often paywalled).

## F. Datasets (papers)
- Jaeger et al., Quant Imaging Med Surg 2014 (PMC4256233) — Montgomery + Shenzhen.
- Rahman et al., IEEE Access 2020 — Qatar/Dhaka TB CXR database.

## Gap statement
The modality-incomplete FL frontier (ClusMFL, FedPIA, feature-imputation) is well-defined
for 2024–2025 but **has not** been applied to the CXR + clinical + sputum TB triad. Fairness
+ heterogeneity for TB FL is likewise underexplored. These two gaps are the project's novelty.
