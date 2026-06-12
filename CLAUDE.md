# CLAUDE.md — Project guide for Claude Code

## What this project is
PhD research on **federated learning (FL) for tuberculosis (TB) detection**. Two objectives:
1. **Heterogeneity- + fairness-aware FL on chest X-rays (CXR)** — the backbone. Simulate
   non-IID clients in Flower; benchmark FedAvg → FedProx → SCAFFOLD (heterogeneity) and
   FairFed / q-FFL (fairness). Primary fairness datasets: **Shenzhen + Montgomery** (only
   public TB CXR sets with per-image age/sex).
2. **Modality-heterogeneous / missing-modality FL** — clients hold different evidence types
   (CXR+demographics, clinical tabular, sputum-smear microscopy). The novel contribution.

The single source of truth for scope/feasibility is
[docs/implementation-plan.md](docs/implementation-plan.md).

## Repository map
- [docs/](docs/) — objectives, datasets, methodology, literature review, research outline,
  implementation plan. **Read these before proposing research changes.**
- [paper/](paper/) — manuscript outline + per-section drafts in `paper/sections/`.
- [code/](code/) — FL implementation (PyTorch + Flower). *(May be empty/rebuilt.)*
- [data/](data/) — dataset prep docs only; **raw data is git-ignored, never commit it.**
- [experiments/](experiments/) — configs (committed), runs (ignored), results (committed).
- [references/](references/) — `references.bib` + per-paper notes.
- [notes/](notes/) — open questions and meeting notes.

## Hard constraints (do not violate)
- **Fairness analysis only on Shenzhen + Montgomery.** TBX11K and Qatar/Dhaka have no
  per-image demographics.
- **Do NOT promise per-patient tri-modal (CXR+clinical+sputum) fusion.** No public dataset
  pairs these per patient. Objective 2 is modality-heterogeneous *across clients*.
- **Sputum** = a separate bacilli-detection modality-client (object detection), not fused
  per-patient features.
- **Never commit raw image data** or large checkpoints (see [.gitignore](.gitignore)).
- Small TB-positive counts (MC 58, SH 336) → **always report 95% CIs**, avoid over-claiming.

## Methods cheat-sheet
- Heterogeneity sim: source-based partitions + Dirichlet label-skew, α ∈ {0.1, 0.5, 1.0, ∞}.
- Robust aggregation: FedProx (primary, tune μ), SCAFFOLD (comparison).
- Fairness: FairFed (group, primary), q-FFL (performance). AIF360 / Fairlearn for metrics.
- Model: DenseNet121 / ResNet, inputs 224/256. XAI: pytorch-grad-cam.
- Rigor: ≥3 seeds, mean ± 95% CI, leave-one-dataset-out external validation.

## Environment & conventions
- **OS:** Windows; shell is **PowerShell** (Bash tool also available).
- **Git:** repo had a "dubious ownership" issue (folder owned by another Windows account) —
  already fixed via `safe.directory`. Author identity is set **repo-locally** (name
  "Dheeraj", email dheeraj.sonkhla15@gmail.com); adjust if needed.
- Commit configs and small result tables; never commit data/checkpoints/logs.
- Compute target is Colab/Kaggle-tier: downsample CXR, subsample TBX11K, use fractional-GPU
  Flower clients. Avoid foundation-model-scale methods (FedPIA) unless compute is secured.

## When unsure
Check [docs/implementation-plan.md](docs/implementation-plan.md) first; record decisions in
[notes/open-questions.md](notes/open-questions.md).
