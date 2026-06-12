# Datasets

Summary of TB datasets relevant to this project. **Raw data is not committed to git** —
this file documents sources, sizes, licenses, and how each is used. See
[../data/README.md](../data/README.md) for download/preparation steps.

## Public TB CXR datasets

| Dataset | Size | Demographics? | Resolution | Use |
|---------|------|---------------|------------|-----|
| **Montgomery County (MC)** | 138 CXR (80 normal / 58 TB) | ✅ age, sex (ClinicalReadings .txt) | ~4020×4892, 12-bit PNG | Fairness backbone |
| **Shenzhen (SH)** | 662 CXR (326 normal / 336 TB) | ✅ age, sex (ClinicalReadings .txt) | ~3000×3000 PNG | Fairness backbone |
| **TBX11K** | 11,200 CXR, bbox TB regions | ❌ | 512×512 | Scale / non-IID client (no fairness) |
| **Qatar/Dhaka (Rahman et al.)** | 3,500 normal + 3,500 TB | ❌ | 512×512 | Extra client (no fairness) |
| **VinDr-CXR** | 18,000 CXR, 28 findings | ✅ age/sex in DICOM tags | DICOM | Non-TB / domain-shift client |
| **IN-CXR (ICMR-NIRT)** | India TB prevalence survey | request-gated | — | India client; leave-one-dataset-out validation |

**Fairness restriction:** only **Shenzhen + Montgomery** carry per-image demographics, so
subgroup fairness analysis is restricted to them. TB-positive counts are small (58 / 336),
so always report confidence intervals.

## Paired CXR + clinical/tabular

- **CODA TB DREAM Challenge** — cough + clinical for 2,143 patients, **no CXR**. Cited as
  motivating evidence that tabular clinical features carry TB signal (clinical-only AUROC
  0.817). Not usable for CXR+clinical fusion.
- **TB Portals (NIAID)** — the only genuinely paired CXR + clinical (+ genomic) TB source
  (~9,020 CXR/CT images). **Gated behind a Data Use Agreement**; skews ~75% drug-resistant.
  → Apply for the DUA early (clinical and imaging are *separate* requests).
- **Verdict:** freely downloadable paired CXR+clinical public TB data does **not** exist →
  adopt the modality-heterogeneous framing for Objective 2.

## Sputum smear microscopy (with bbox/XML)

- **Kaggle "Tuberculosis Image Dataset" (saife245)** — 928 images, 3,734 bacilli instances,
  bounding-box annotations. Closest public match to the project's ~1000-image asset.
  **TODO: confirm XML schema is PASCAL VOC** (`<annotation><object><bndbox>`).
- **ZNSM-iDB** — ~2,000 Ziehl-Neelsen images, segmentation-style annotations.
- **Costa/TBimages (IEEE DataPort)** — 120 annotated images from 12 patients.
- **Gomide et al. 2025** — 502 annotated ZN images + per-bacillus positions + code.

**Critical:** sputum datasets share **no patients** with any CXR dataset (different
modality, geography, institution). Per-patient CXR+sputum fusion is impossible with public
data; combining across *clients* is legitimate (not patient leakage).

## The project's own sputum asset
- ~1000 sputum smear images + XML annotations.
- **Action items:** (1) inspect one XML file, confirm format; (2) treat as a separate
  bacilli-detection modality-client (object detection), not fused per-patient features.

## Open data tasks
- [ ] Download Montgomery + Shenzhen; parse `ClinicalReadings` demographics.
- [ ] Apply for TB Portals DUA.
- [ ] Confirm sputum XML schema and source.
- [ ] Decide TBX11K subsampling strategy (compute budget).
