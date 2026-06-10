# System Architecture

## Pipeline Overview

```
                            ┌──────────────────────────────────────┐
                            │   Onboard Sensors                    │
                            │   - Camera (downward)                │
                            │   - IMU (accelerometer + magnetometer)│
                            │   - Barometer (optional)             │
                            │   - GPS (when available)             │
                            └─────────┬────────────────────────────┘
                                      │
                                      ▼
                            ┌──────────────────────┐
                            │   UAV Image (224×224)│
                            └─────────┬────────────┘
                                      │
                                      ▼
              ┌─────────────────────────────────────────────────────────┐
              │   Stage 1: Visual Retrieval (MFFT ensemble)             │
              │   Output: Top-K candidate satellite tiles from database │
              └─────────┬───────────────────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────────────────────────────────────────────┐
              │   Stage 2: IMU Prior Re-ranking (Mode D)                │
              │   Score = sim - λ · (d / σ)²                            │
              │   Output: Top-1 satellite tile (after altitude filter)  │
              └─────────┬───────────────────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────────────────────────────────────────────┐
              │   Stage 3: Pose Refinement Head ◄── this demo           │
              │   Input: (UAV image, matched satellite tile)            │
              │   Output: Δx (east), Δy (north), Z bucket, ψ confidence │
              └─────────┬───────────────────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────────────────────────────────────────────┐
              │   Stage 4: Sensor Fusion (ESKF15)                       │
              │   Combines visual update with IMU integration           │
              │   Output: Final (x, y, z, ψ) at 5-50 Hz                 │
              └─────────────────────────────────────────────────────────┘
```

## Pose Refinement Head (this demo)

A shared-backbone Siamese ResNet18, adapted from Pratima et al.
(SS-BEVR-UAR, 2026) with three modifications:

| Aspect | Source paper | This work |
|--------|--------------|-----------|
| Input modality | LiDAR BEV | UAV RGB + satellite RGB |
| Backbone | 4-layer stride-2 conv (from scratch, ~1.5 M params) | Shared ResNet18 (ImageNet pretrained, ~12 M params) |
| Output | (Δx, Δy, Δθ) + σ | (Δx, Δy) regression + Z 4-class classification |

```
UAV image (3×224×224) ─┐
                       ├─► Shared ResNet18 ──► 512-d UAV feature
Sat tile (3×224×224) ──┘                    ──► 512-d Sat feature

fused = concat(UAV_feat, Sat_feat, |UAV − Sat|)   (1536-d)
              │
              ▼
        MLP (1536 → 512 → 256)
              │
   ┌──────────┼──────────────┐
   ▼          ▼              ▼
 head_xy   head_yaw       head_z
(Gaussian) (Gaussian)   (4-class CE)
```

- ImageNet normalization is applied inside `forward()`; raw [0, 1] tensors are
  expected as input.
- Training uses augmentation of ±5 m satellite shift + ±180° UAV rotation.
- Loss: Gaussian NLL for `xy`, cross-entropy for `z`.

## Edge Deployment

The full pipeline runs on NVIDIA Jetson AGX Xavier 32GB:

| Stage | Latency (FP16) |
|-------|----------------|
| Image preprocessing | ~2 ms |
| MFFT feature extraction (8 TTA views) | ~22 ms |
| IMU prior re-ranking | ~1 ms |
| Pose refinement head | ~3 ms |
| **End-to-end** | **~28 ms (≈ 35 FPS)** |

Closed-loop control runs the vision update at 5 Hz, fused with 200 Hz IMU via
an Error-State Kalman Filter (ESKF15) implementation.
