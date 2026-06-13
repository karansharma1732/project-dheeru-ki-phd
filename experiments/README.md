# Experiments

Track experiment configs, runs, and results here. Keep raw outputs git-ignored; commit
configs and small summary tables.

## Layout
```
experiments/
├── configs/      # one YAML per experiment (committed)
├── runs/         # logs, checkpoints, metrics (git-ignored)
└── results/      # summary CSVs / tables for the paper (committed)
```

## Run log

| ID | Date | Method | Partition (α) | Datasets | Seeds | Key metric | Notes |
|----|------|--------|---------------|----------|-------|-----------|-------|
| ex000 | 2026-06-13 | centralized DenseNet121 | — | SH+MC | 1 | test AUC=0.960 | acc=0.892 sens=0.814 spec=0.967 F1=0.881 (n=120). Fairness gaps: sex sens 0.69(F)/0.86(M); age worst 60+ AUC=0.85. Single seed — needs ≥3 seeds + 95% CIs. |

## Conventions
- Experiment ID: `exNNN`.
- Always log: method, partition/α, datasets, seeds, rounds, LR, and final metrics.
- Reproducibility: pin seed, dataset version, and commit hash in each config.
