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
| _ex000_ | | centralized | — | SH+MC | 3 | AUC= | baseline template |

## Conventions
- Experiment ID: `exNNN`.
- Always log: method, partition/α, datasets, seeds, rounds, LR, and final metrics.
- Reproducibility: pin seed, dataset version, and commit hash in each config.
