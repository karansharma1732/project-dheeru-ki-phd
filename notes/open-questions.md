# Open Questions & Decisions

## Open questions
- [ ] Is the sputum XML PASCAL VOC? (inspect one file)
- [ ] TB Portals DUA — apply now; clinical and imaging are separate requests.
- [ ] TBX11K subsampling strategy under Colab compute budget?
- [ ] Confirm HPFL / FedARC methods from primary venues for differentiation.
- [ ] Encoder choice: DenseNet121 vs ResNet — settle in Phase 1.

## Decisions log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-12 | Reframe Objective 2 as modality-heterogeneous FL | No public dataset pairs CXR+clinical+sputum per patient |
| 2026-06-12 | Restrict fairness analysis to Shenzhen + Montgomery | Only sets with per-image demographics |
