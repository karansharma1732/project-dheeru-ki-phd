# ex000 — Centralized DenseNet121 baseline

Paper-ready results for the centralized upper-bound baseline. Source run:
`MyDrive/tb-fl/experiments/runs/ex000_centralized/test_metrics.json` (Drive).

- **Model:** DenseNet121, ImageNet-pretrained, 224×224 input
- **Data:** Shenzhen + Montgomery (800 images; 394 TB / 406 normal)
- **Split:** stratified train 560 / val 120 / test 120
- **Training:** 25 epochs max, early-stopped at epoch 20 (best val AUC 0.950 @ epoch 14)
- **Seeds:** 1 (seed=42) — ⚠ needs ≥3 seeds + 95% CIs before paper claims
- **Date:** 2026-06-13

## Overall (test, n=120)

| Metric | Value |
|--------|-------|
| AUC | 0.960 |
| Accuracy | 0.892 |
| Sensitivity | 0.814 |
| Specificity | 0.967 |
| F1 | 0.881 |

## Fairness — by sex

| Group | n | AUC | Sensitivity |
|-------|---|-----|-------------|
| Male | 78 | 0.977 | 0.860 |
| Female | 40 | 0.909 | 0.688 |
| unknown | 2 | — | — |
| **AUC gap** | | 0.068 | |
| **Sensitivity gap** | | | 0.173 |

## Fairness — by age band

| Group | n | AUC | Sensitivity |
|-------|---|-----|-------------|
| 45–60 | 17 | 1.000 | 1.000 |
| 30–45 | 46 | 0.962 | 0.808 |
| 0–30 | 44 | 0.950 | 0.824 |
| 60+ | 13 | 0.850 | 0.625 |
| **AUC gap** | | 0.150 | |
| **Sensitivity gap** | | | 0.375 |

## Notes
- Headline disparities: female sensitivity 0.69 vs male 0.86; 60+ group worst (AUC 0.85,
  sensitivity 0.63). These motivate the fairness objective but are single-seed on small
  subgroups (F n=40, 60+ n=13) — directional only.
- Per-subgroup specificity/F1/accuracy are in the full `test_metrics.json` in Drive (only
  the per-group AUC/sensitivity were captured to this table).
