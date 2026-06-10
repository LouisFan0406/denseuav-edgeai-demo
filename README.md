# UAV Visual Self-Localization on Edge Devices

> **NTUST EdgeAI Final Project** (EE5354701)
> **Author:** 范詩涵 Shih-Han Fan (M11407W19)
> **Advisor:** Prof. Chih-Lyang Hwang
> **Year:** 2026

A real-time UAV self-localization system deployable on NVIDIA Jetson AGX Xavier.
The full pipeline combines visual retrieval (MFFT), IMU prior re-ranking, and a
partial 3D pose refinement head.

This repository provides a **minimal, verifiable inference demo** for the
partial 3D pose head. Full training code, dataset, and best-performing model
weights are kept private due to ongoing research.

---

## Key Results

| Metric | Value |
|--------|-------|
| Retrieval R@1 (cross-area, HUAZHONG zero-shot) | **96.78%** |
| End-to-end mean position error (synthetic 5 m offset) | **0.77 m** |
| AGX Xavier TensorRT FP16 latency | **26.7 ms** |
| AGX Xavier throughput | **37.5 FPS** |
| Cross-platform accuracy delta (PC vs AGX) | **±1.3%** |
| Closed-loop simulator (vision + ESKF15) | **4.58 Hz vision, σ ≈ 1.1 m** |

Details: see [docs/results.md](docs/results.md).

---

## Demo Video

[YouTube demo — 8-10 minute walkthrough](https://www.youtube.com/watch?v=EkgIz9K4e2w)

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/LouisFan0406/denseuav-edgeai-demo
cd denseuav-edgeai-demo
pip install -r requirements.txt
```

### 2. Obtain the demo checkpoint and sample images

See [checkpoints/README.md](checkpoints/README.md) and
[sample_data/README.md](sample_data/README.md) for how to obtain the demo files.

### 3. Run inference on the sample pair

```bash
python inference_demo.py
```

Expected output:

```
============================================================
  Partial 3D Pose Estimation Result
============================================================
  Translation (ENU):
    Δx (east)  = +0.182 m
    Δy (north) = -0.094 m

  Altitude (4-class):
    Predicted bucket = 50 m
    Probabilities    = {40m: 0.071, 50m: 0.812, 60m: 0.105, 80m: 0.012}

  Inference latency (median of 10 runs): 28.45 ms
============================================================
```

(Actual numbers depend on your hardware and the specific sample images.)

### 4. Run on your own image pair

```bash
python inference_demo.py \
    --uav path/to/your/uav_image.jpg \
    --sat path/to/your/satellite_tile.jpg
```

---

## What's in this Repository

```
denseuav-edgeai-demo/
├── inference_demo.py            # Main entry point
├── src/
│   └── pose_refinement_head_resnet.py   # Model definition (ResNet18 Siamese)
├── sample_data/                 # Demo input pair + expected output
├── checkpoints/                 # Demo model weight (request access)
├── docs/
│   ├── architecture.md          # System overview and design decisions
│   └── results.md               # Detailed evaluation metrics
├── requirements.txt
├── LICENSE                      # MIT (for this demo code only)
└── README.md
```

---

## What's NOT in this Repository

- Training scripts and training data (proprietary)
- Full MFFT retrieval models (ensemble v8 + v25 + v29)
- Closed-loop ESKF15 simulator
- AGX deployment scripts and TensorRT engine builds

If you need access for academic collaboration, please contact the author.

---

## Citation

If you use this work or its ideas, please cite:

```bibtex
@misc{fan2026denseuav,
  author       = {Fan, Shih-Han},
  title        = {UAV Visual Self-Localization on Edge Devices:
                  Retrieval, IMU Prior Re-ranking, and Partial 3D Pose Refinement},
  howpublished = {NTUST EE5354701 Edge AI Final Project},
  year         = {2026},
}
```

---

## Contact

范詩涵 Shih-Han Fan
National Taiwan University of Science and Technology (NTUST)
ledlab2392@gmail.com
