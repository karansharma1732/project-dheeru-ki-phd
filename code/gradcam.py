"""Grad-CAM sanity check — overlay TB saliency on a few CXRs.

Verifies the classifier attends to lung regions rather than artefacts. Uses the
pytorch-grad-cam package (``pip install grad-cam``).

Usage:
    python gradcam.py --config config.yaml --checkpoint <run>/best.pt --n 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from data_loader import build_transforms
from models import build_model, gradcam_target_layer

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install grad-cam: pip install grad-cam") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    size = cfg["data"]["image_size"]

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    target_layer = gradcam_target_layer(model, cfg["model"]["arch"])
    cam = GradCAM(model=model, target_layers=[target_layer])

    out_dir = Path(cfg["paths"]["out_dir"]) / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["paths"]["demographics_csv"])
    tb = df[(df["label"] == 1) & (df["image_path"].astype(str).str.len() > 0)]
    tf = build_transforms(size, train=False)

    for _, row in tb.head(args.n).iterrows():
        img = Image.open(row["image_path"]).convert("L")
        x = tf(img).unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
        rgb = np.array(img.convert("RGB").resize((size, size))) / 255.0
        overlay = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)
        Image.fromarray(overlay).save(out_dir / f"{row['filename']}_cam.png")

    print(f"[done] Grad-CAM overlays -> {out_dir}")


if __name__ == "__main__":
    main()
