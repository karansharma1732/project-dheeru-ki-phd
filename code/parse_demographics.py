"""Parse Montgomery + Shenzhen ClinicalReadings into a single demographics CSV.

Both NLM TB datasets ship per-image ``ClinicalReadings/*.txt`` files containing the
patient's age and sex, plus a free-text reading. The binary label is encoded in the
image filename: the trailing ``_0`` means normal and ``_1`` means abnormal/TB
(e.g. ``CHNCXR_0001_0.png``, ``MCUCXR_0350_1.png``).

The two datasets use slightly different text layouts, so we extract age and sex with
tolerant regexes rather than fixed line positions.

Output columns: dataset, image_path, filename, label, sex, age, age_band.

Run (no GPU needed):
    python parse_demographics.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import yaml

# Trailing _0 / _1 before the extension encodes the label.
_LABEL_RE = re.compile(r"_(\d)\.(?:png|txt)$", re.IGNORECASE)
_SEX_RE = re.compile(r"\b(male|female|man|woman)\b|sex\s*[:\-]?\s*([mf])\b", re.IGNORECASE)
_AGE_RE = re.compile(r"(\d{1,3})\s*(?:y(?:ear|r)?s?|y\b)", re.IGNORECASE)
# Fallback: "Patient's Age: 044Y" style or a lone number near the word "age".
_AGE_FALLBACK_RE = re.compile(r"age\s*[:\-]?\s*0*(\d{1,3})", re.IGNORECASE)


def label_from_filename(name: str) -> int | None:
    m = _LABEL_RE.search(name)
    return int(m.group(1)) if m else None


def parse_sex(text: str) -> str:
    m = _SEX_RE.search(text)
    if not m:
        return "unknown"
    token = (m.group(1) or m.group(2) or "").lower()
    if token in {"m", "male", "man"}:
        return "M"
    if token in {"f", "female", "woman"}:
        return "F"
    return "unknown"


def parse_age(text: str) -> int | None:
    for rx in (_AGE_RE, _AGE_FALLBACK_RE):
        m = rx.search(text)
        if m:
            age = int(m.group(1))
            if 0 < age < 120:
                return age
    return None


def age_to_band(age: int | None, bands: list[int]) -> str:
    if age is None:
        return "unknown"
    for lo, hi in zip(bands[:-1], bands[1:]):
        if lo <= age < hi:
            return f"{lo}-{hi}"
    return "unknown"


def find_image(clinical_txt: Path) -> Path | None:
    """Locate the CXR image matching a ClinicalReadings .txt file."""
    stem = clinical_txt.stem
    # Images usually live in a sibling CXR_png/ folder; search the dataset root too.
    root = clinical_txt.parent.parent
    for ext in (".png", ".jpg", ".jpeg"):
        for cand in root.rglob(stem + ext):
            return cand
    return None


def parse_dataset(name: str, root: Path, bands: list[int]) -> list[dict]:
    root = Path(root)
    clinical_dir = next((p for p in root.rglob("ClinicalReadings")), None)
    if clinical_dir is None:
        print(f"[warn] no ClinicalReadings/ under {root} — skipping {name}")
        return []

    rows: list[dict] = []
    for txt in sorted(clinical_dir.glob("*.txt")):
        label = label_from_filename(txt.name)
        if label is None:
            print(f"[warn] no label in filename {txt.name} — skipping")
            continue
        text = txt.read_text(encoding="utf-8", errors="ignore")
        age = parse_age(text)
        img = find_image(txt)
        rows.append(
            {
                "dataset": name,
                "image_path": str(img) if img else "",
                "filename": txt.stem,
                "label": label,
                "sex": parse_sex(text),
                "age": age if age is not None else "",
                "age_band": age_to_band(age, bands),
            }
        )
    print(f"[ok] {name}: parsed {len(rows)} records")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    bands = cfg["data"]["age_bands"]
    out_csv = Path(cfg["paths"]["demographics_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ds in cfg["data"]["datasets"]:
        rows += parse_dataset(ds, cfg["paths"][ds], bands)

    if not rows:
        raise SystemExit("No records parsed — check dataset paths in config.yaml.")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_tb = sum(r["label"] == 1 for r in rows)
    n_unknown_sex = sum(r["sex"] == "unknown" for r in rows)
    n_unknown_age = sum(r["age_band"] == "unknown" for r in rows)
    print(f"[done] wrote {len(rows)} rows -> {out_csv}")
    print(f"       TB-positive: {n_tb} | normal: {len(rows) - n_tb}")
    print(f"       unknown sex: {n_unknown_sex} | unknown age: {n_unknown_age}")


if __name__ == "__main__":
    main()
