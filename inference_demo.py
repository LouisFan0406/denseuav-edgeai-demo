"""UAV Visual Self-Localization — Inference Demo.

Loads a pre-trained pose refinement head and runs partial 3D pose estimation
on a (UAV image, satellite tile) pair.

Outputs:
    - Δx, Δy translation estimate (meters, ENU)
    - Altitude (4-class: 40 / 50 / 60 / 80 m)
    - Inference latency

Usage:
    python inference_demo.py
    python inference_demo.py --uav path/to/uav.jpg --sat path/to/sat.jpg
    python inference_demo.py --ckpt checkpoints/pose_head_demo.pth

Verification:
    Run on the provided sample_data and compare with sample_data/expected_output.json.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.pose_refinement_head_resnet import PoseRefinementHeadResNet18

ALTITUDE_BUCKETS = {0: 40, 1: 50, 2: 60, 3: 80}
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "pose_head_demo.pth"
DEFAULT_UAV = REPO_ROOT / "sample_data" / "uav_sample.jpg"
DEFAULT_SAT = REPO_ROOT / "sample_data" / "sat_sample.jpg"
IMAGE_SIZE = 224


def load_image(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    return TF.to_tensor(img).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UAV pose refinement demo")
    p.add_argument("--uav", type=Path, default=DEFAULT_UAV)
    p.add_argument("--sat", type=Path, default=DEFAULT_SAT)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Device:     {args.device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"UAV image:  {args.uav}")
    print(f"Sat tile:   {args.sat}")
    print()

    if not args.ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        print("See checkpoints/README.md for how to obtain the demo checkpoint.")
        sys.exit(1)
    if not args.uav.exists() or not args.sat.exists():
        print("[ERROR] Sample images not found.")
        print("See sample_data/README.md for how to obtain the sample images.")
        sys.exit(1)

    model = PoseRefinementHeadResNet18(
        latent_dim=256,
        num_altitude_classes=4,
        pretrained=False,
        dropout=0.0,
    ).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    uav = load_image(args.uav, IMAGE_SIZE).to(args.device)
    sat = load_image(args.sat, IMAGE_SIZE).to(args.device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(uav, sat)
    if args.device == "cuda":
        torch.cuda.synchronize()

    # Timed inference (10 runs, report median)
    times = []
    with torch.no_grad():
        for _ in range(10):
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(uav, sat)
            if args.device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    dx = float(out["mu_xy"][0, 0].item())
    dy = float(out["mu_xy"][0, 1].item())
    z_idx = int(out["z_logits"].argmax(dim=1).item())
    z_m = ALTITUDE_BUCKETS[z_idx]
    z_probs = torch.softmax(out["z_logits"], dim=1)[0].cpu().tolist()
    times.sort()
    median_ms = times[len(times) // 2]

    print("=" * 60)
    print("  Partial 3D Pose Estimation Result")
    print("=" * 60)
    print(f"  Translation (ENU):")
    print(f"    Δx (east)  = {dx:+.3f} m")
    print(f"    Δy (north) = {dy:+.3f} m")
    print()
    print(f"  Altitude (4-class):")
    print(f"    Predicted bucket = {z_m} m")
    print(f"    Probabilities    = {{40m: {z_probs[0]:.3f}, "
          f"50m: {z_probs[1]:.3f}, 60m: {z_probs[2]:.3f}, 80m: {z_probs[3]:.3f}}}")
    print()
    print(f"  Inference latency (median of 10 runs): {median_ms:.2f} ms")
    print("=" * 60)

    expected = REPO_ROOT / "sample_data" / "expected_output.json"
    if expected.exists() and args.uav == DEFAULT_UAV and args.sat == DEFAULT_SAT:
        with open(expected, encoding="utf-8") as f:
            ref = json.load(f)
        print()
        print("  Sanity check against expected_output.json:")
        print(f"    Expected dx ≈ {ref['dx_m']:+.3f} m   (got {dx:+.3f} m)")
        print(f"    Expected dy ≈ {ref['dy_m']:+.3f} m   (got {dy:+.3f} m)")
        print(f"    Expected Z  = {ref['z_m']} m         (got {z_m} m)")


if __name__ == "__main__":
    main()
