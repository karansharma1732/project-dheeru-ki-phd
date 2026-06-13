"""Phase 1 — train the centralized TB classifier on Shenzhen + Montgomery.

This is the upper-bound baseline and, more importantly, it locks in the metric harness
(metrics.evaluate) that every federated experiment in Phase 2+ will reuse.

The training+eval for a single seed lives in ``run_once`` so the multi-seed runner
(run_multiseed.py) can reuse it.

Usage (Colab GPU):
    python train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from data_loader import build_dataloaders
from metrics import evaluate, format_report
from models import build_model


def resolve_device(cfg: dict) -> str:
    return cfg["device"] if (torch.cuda.is_available() or cfg["device"] == "cpu") else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(cfg, model):
    p = model.parameters()
    lr, wd = cfg["train"]["lr"], cfg["train"]["weight_decay"]
    if cfg["train"]["optimizer"].lower() == "adam":
        return torch.optim.Adam(p, lr=lr, weight_decay=wd)
    return torch.optim.SGD(p, lr=lr, weight_decay=wd, momentum=0.9)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def run_once(cfg: dict, seed: int, out_dir: Path, device: str) -> dict:
    """Train + evaluate a single seed. Returns the test-metrics dict.

    The data split is seed-dependent (data_loader reads cfg['seed']), so each seed
    yields a fresh train/val/test partition — this captures split + init variance for CIs.
    """
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_w = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_w.to(device) if cfg["train"]["use_class_weights"] else None
    )
    optimizer = make_optimizer(cfg, model)
    attrs = tuple(cfg["eval"]["sensitive_attrs"])

    best_auc, patience, history = -1.0, 0, []
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, device, attrs)
        val_auc = val["overall"]["auc"]
        history.append({"epoch": epoch, "train_loss": loss, "val": val})
        print(f"  [seed {seed}] epoch {epoch:02d} | train_loss={loss:.4f} | val_auc={val_auc:.3f}")

        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print(f"  [seed {seed}] early stop — no val AUC gain for {patience} epochs")
                break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test = evaluate(model, test_loader, device, attrs)
    (out_dir / "test_metrics.json").write_text(json.dumps(test, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = resolve_device(cfg)
    out_dir = Path(cfg["paths"]["out_dir"])
    test = run_once(cfg, cfg["seed"], out_dir, device)
    print("\n=== TEST ===\n" + format_report(test))
    print(f"\n[done] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
