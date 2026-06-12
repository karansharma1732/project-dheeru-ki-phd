# Data

**Raw data and large files are NOT committed to git** (see [.gitignore](../.gitignore)).
This folder holds download/preparation instructions and small derived metadata only.

## Suggested local layout (git-ignored)
```
data/
├── raw/
│   ├── montgomery/        # MontgomerySet: CXR_png/, ClinicalReadings/, ManualMask/
│   ├── shenzhen/          # ChinaSet_AllFiles: CXR_png/, ClinicalReadings/
│   ├── tbx11k/
│   ├── qatar/
│   └── sputum/            # ~1000 images + XML (PASCAL VOC?)
├── processed/             # resized CXR, harmonized demographics CSV, partitions
└── splits/                # client partitions, seeds, leave-one-dataset-out folds
```

## Datasets & sources
See [../docs/datasets.md](../docs/datasets.md) for full details, sizes, and licenses.

| Dataset | Source |
|---------|--------|
| Montgomery + Shenzhen | NLM (Jaeger et al. 2014) |
| TBX11K | mmcheng.net/tb |
| Qatar/Dhaka | Kaggle (Rahman et al.) |
| Sputum (Kaggle saife245) | Kaggle "Tuberculosis Image Dataset" |
| TB Portals | NIAID (Data Use Agreement required) |

## Preparation tasks
- [ ] Download Montgomery + Shenzhen; verify `ClinicalReadings/*.txt`.
- [ ] Write a parser to extract (age, sex, label) → `processed/demographics.csv`.
- [ ] Resize CXR to 224/256; record preprocessing params.
- [ ] Inspect one sputum XML; confirm PASCAL VOC schema.
- [ ] Generate client partitions (source-based + Dirichlet α sweep) into `splits/`.

## Provenance & ethics
Record per-dataset license and any data-use restrictions before publishing results.
