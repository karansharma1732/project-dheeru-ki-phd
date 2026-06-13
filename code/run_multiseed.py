"""Phase 1 — multi-seed runner with 95% confidence intervals.

Runs the centralized baseline over several seeds and aggregates each metric as
mean ± 95% CI (Student-t, appropriate for the small number of seeds). Emits:

  <out_dir>/seed<k>/...            per-seed checkpoint + metrics (best.pt git-ignored)
  <out_dir>/aggregate.json         structured mean/CI for every metric and subgroup
  <out_dir>/summary.csv            tidy, paper-ready table (one row per metric/group)
  <out_dir>/summary.md             human-readable summary tables

Download <out_dir> into experiments/results/ and commit the json/csv/md (the *.pt
checkpoints stay git-ignored), exactly as for the single-seed run.

Usage (Colab GPU):
    python run_multiseed.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from metrics import format_report
from train import resolve_device, run_once

try:
    from scipy.stats import t as student_t

    def _t_crit(n: int) -> float:
        return float(student_t.ppf(0.975, n - 1))
except ImportError:  # scipy unavailable — fall back to a small lookup table
    _T_TABLE = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}

    def _t_crit(n: int) -> float:
        return _T_TABLE.get(n, 1.96)


def mean_ci(values: list[float]) -> dict:
    """Mean and 95% t-CI over the finite values in the list."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "ci95": float("nan"), "std": 0.0, "n": 1}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    ci95 = _t_crit(n) * std / math.sqrt(n)
    return {"mean": mean, "ci95": ci95, "std": std, "n": n}


# Metrics to aggregate at each scope.
_OVERALL_KEYS = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
_GROUP_KEYS = ["auc", "sensitivity", "specificity", "f1"]
_GAP_KEYS = ["auc_gap", "sensitivity_gap", "worst_group_auc"]


def aggregate(results: list[dict], attrs: list[str]) -> dict:
    """Collapse a list of per-seed test-metric dicts into mean/CI structures."""
    agg: dict = {"seeds": len(results), "overall": {}, "subgroups": {}}

    for k in _OVERALL_KEYS:
        agg["overall"][k] = mean_ci([r["overall"][k] for r in results])

    for attr in attrs:
        # Union of group labels seen across seeds.
        labels = sorted({g for r in results for g in r["subgroups"][attr]["groups"]})
        group_stats = {}
        for label in labels:
            group_stats[label] = {
                k: mean_ci(
                    [
                        r["subgroups"][attr]["groups"][label][k]
                        for r in results
                        if label in r["subgroups"][attr]["groups"]
                    ]
                )
                for k in _GROUP_KEYS
            }
        gaps = {k: mean_ci([r["subgroups"][attr][k] for r in results]) for k in _GAP_KEYS}
        agg["subgroups"][attr] = {"groups": group_stats, "gaps": gaps}
    return agg


def _fmt(stat: dict) -> str:
    if stat["n"] == 0 or math.isnan(stat["mean"]):
        return "—"
    if stat["n"] == 1 or math.isnan(stat["ci95"]):
        return f"{stat['mean']:.3f}"
    return f"{stat['mean']:.3f} ± {stat['ci95']:.3f}"


def write_csv(agg: dict, path: Path) -> None:
    rows = [("scope", "group", "metric", "mean", "ci95", "std", "n_seeds")]
    for k, s in agg["overall"].items():
        rows.append(("overall", "-", k, s["mean"], s["ci95"], s["std"], s["n"]))
    for attr, data in agg["subgroups"].items():
        for label, metrics in data["groups"].items():
            for k, s in metrics.items():
                rows.append((attr, label, k, s["mean"], s["ci95"], s["std"], s["n"]))
        for k, s in data["gaps"].items():
            rows.append((attr, "ALL", k, s["mean"], s["ci95"], s["std"], s["n"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(
                [r[0], r[1], r[2]]
                + [f"{v:.6f}" if isinstance(v, float) else v for v in r[3:]]
            )


def write_md(agg: dict, path: Path, exp_name: str) -> None:
    n = agg["seeds"]
    lines = [
        f"# {exp_name} — multi-seed results ({n} seeds, mean ± 95% CI)",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k in _OVERALL_KEYS:
        lines.append(f"| {k} | {_fmt(agg['overall'][k])} |")
    for attr, data in agg["subgroups"].items():
        lines += ["", f"## By {attr}", "", "| Group | AUC | Sensitivity | Specificity | F1 |",
                  "|-------|-----|-------------|-------------|----|"]
        for label, m in data["groups"].items():
            lines.append(
                f"| {label} | {_fmt(m['auc'])} | {_fmt(m['sensitivity'])} | "
                f"{_fmt(m['specificity'])} | {_fmt(m['f1'])} |"
            )
        g = data["gaps"]
        lines += [
            "",
            f"- **AUC gap:** {_fmt(g['auc_gap'])} | **Sensitivity gap:** "
            f"{_fmt(g['sensitivity_gap'])} | **Worst-group AUC:** {_fmt(g['worst_group_auc'])}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    ms = cfg.get("multiseed", {})
    seeds = ms.get("seeds", [0, 1, 2])
    exp_name = ms.get("exp_name", "ex001_centralized_multiseed")
    device = resolve_device(cfg)
    attrs = list(cfg["eval"]["sensitive_attrs"])

    base = Path(cfg["paths"]["out_dir"]).parent / exp_name
    base.mkdir(parents=True, exist_ok=True)
    print(f"[multiseed] {exp_name} | seeds={seeds} | device={device}\n")

    results = []
    for seed in seeds:
        print(f"=== seed {seed} ===")
        res = run_once(cfg, seed, base / f"seed{seed}", device)
        results.append(res)
        print(f"  -> test AUC={res['overall']['auc']:.3f}\n")

    agg = aggregate(results, attrs)
    (base / "aggregate.json").write_text(json.dumps(agg, indent=2))
    write_csv(agg, base / "summary.csv")
    write_md(agg, base / "summary.md", exp_name)

    print("=== AGGREGATE (mean ± 95% CI) ===")
    print(f"overall AUC: {_fmt(agg['overall']['auc'])}")
    for attr in attrs:
        g = agg["subgroups"][attr]["gaps"]
        print(f"[{attr}] sens_gap={_fmt(g['sensitivity_gap'])} auc_gap={_fmt(g['auc_gap'])}")
    print(f"\n[done] artifacts -> {base}")
    print("       commit: summary.csv, summary.md, aggregate.json, seed*/test_metrics.json")


if __name__ == "__main__":
    main()
