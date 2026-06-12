"""Evaluation + fairness metrics — the harness every later experiment reuses.

``evaluate`` returns overall metrics (accuracy, sensitivity, specificity, AUC, F1)
plus per-subgroup metrics and fairness gaps for each sensitive attribute. Keep this
stable: Phase 2+ (federated) calls the exact same function so results stay comparable.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # recall / TPR
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    # AUC / F1 need both classes present.
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "n": n,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "auc": auc,
        "f1": f1,
    }


def _subgroup_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, groups: np.ndarray
) -> dict:
    """Per-group metrics + gaps (max - min) across groups for AUC/sensitivity."""
    out: dict = {"groups": {}}
    aucs, senss = {}, {}
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        if mask.sum() == 0:
            continue
        m = _binary_metrics(y_true[mask], y_pred[mask], y_prob[mask])
        out["groups"][str(g)] = m
        if not np.isnan(m["auc"]):
            aucs[g] = m["auc"]
        if not np.isnan(m["sensitivity"]):
            senss[g] = m["sensitivity"]

    def gap(d: dict) -> float:
        return (max(d.values()) - min(d.values())) if len(d) > 1 else float("nan")

    out["auc_gap"] = gap(aucs)
    out["sensitivity_gap"] = gap(senss)
    # Worst-group performance — a key fairness headline.
    out["worst_group_auc"] = min(aucs.values()) if aucs else float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device, sensitive_attrs=("sex", "age_band")) -> dict:
    model.eval()
    probs, preds, trues, metas = [], [], [], []
    for x, y, meta in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(TB)
        probs.append(p)
        preds.append((p >= 0.5).astype(int))
        trues.append(y.numpy())
        metas.extend(meta)

    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

    result = {"overall": _binary_metrics(y_true, y_pred, y_prob), "subgroups": {}}
    for attr in sensitive_attrs:
        groups = np.array([str(m.get(attr, "unknown")) for m in metas])
        result["subgroups"][attr] = _subgroup_metrics(y_true, y_pred, y_prob, groups)
    return result


def format_report(result: dict) -> str:
    o = result["overall"]
    lines = [
        f"overall (n={o['n']}): acc={o['accuracy']:.3f} sens={o['sensitivity']:.3f} "
        f"spec={o['specificity']:.3f} auc={o['auc']:.3f} f1={o['f1']:.3f}",
    ]
    for attr, sg in result["subgroups"].items():
        lines.append(
            f"  [{attr}] auc_gap={sg['auc_gap']:.3f} "
            f"sens_gap={sg['sensitivity_gap']:.3f} "
            f"worst_group_auc={sg['worst_group_auc']:.3f}"
        )
        for g, m in sg["groups"].items():
            lines.append(f"      {g}: n={m['n']} auc={m['auc']:.3f} sens={m['sensitivity']:.3f}")
    return "\n".join(lines)
