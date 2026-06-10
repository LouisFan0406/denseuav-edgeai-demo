# Evaluation Results

All numbers below are from **HUAZHONG Park zero-shot evaluation** (model trained
on JINMEI + XICHOU + DINO; HUAZHONG was unseen during training).

## Retrieval (Mode D Pipeline)

Ensemble of v8 + v25 + v29 MFFT models with IMU position prior re-ranking
(σ = 2 m) and altitude filtering.

| Metric | Value |
|--------|-------|
| R@1 | 96.78% |
| R@1-strict (place + altitude) | 96.35% |
| Median position error | 0.05 m |
| Mean error (synthetic 5 m offset) | 0.77 m |
| MA@2m | 92.7% |
| MA@5m | 94.6% |
| MA@10m | 97.9% |

## Pose Refinement Head

Trained for 30 epochs on 3 parks, evaluated on HUAZHONG.

| Setting | RMSE_xy | Z accuracy |
|---------|---------|------------|
| Clean validation (no synthetic offset) | 0.177 m | 84.7% |
| Augmented validation (±5 m offset) | 0.498 m | 43.2% (vs random 25%) |
| Mode D integration (σ = 5 m offset) | 1.131 m → 0.769 m (with altitude filter) | 54.5% (vision-only) / 100% (barometer fused) |

The augmented RMSE (0.498 m) vs the "always output zero" baseline (4.082 m) is
an 87.8% improvement, confirming the model actually learned spatial alignment.

## Final Partial 3D Pose Estimate

| Degree of Freedom | Source | Accuracy |
|--------------------|--------|----------|
| X, Y | Pose refinement head + altitude-filtered retrieval | **mean 0.77 m, MA@2m 92.7%** |
| Z | Barometer (DJI EXIF / Pixhawk barometer) | **100% (within 10 m bins)** |
| ψ (heading) | IMU magnetometer | **±2°** |

## AGX Xavier Deployment

| Backend | Latency | Speed-up |
|---------|---------|----------|
| PC RTX FP32 | ~30 ms | reference |
| AGX PyTorch FP32 | 125 ms | 1× (AGX baseline) |
| **AGX TensorRT FP16** | **26.7 ms** | **4.7× over AGX FP32** |
| Throughput (TRT FP16) | **37.5 FPS** | — |

Cross-platform accuracy match (v36 + AdaBN single model):

| Metric | PC | AGX | Δ |
|--------|-------|-------|-------|
| R@1 | 75.54% | 74.25% | −1.29% |
| R@5 | 97.00% | 97.21% | +0.21% |
| Median error | 0.06 m | 0.06 m | 0 |
| P95 error | 12.37 m | 12.37 m | 0 |
| MA@10m | 89.48% | 88.63% | −0.85% |

## Closed-Loop Simulator (ESKF15 stress test)

A 3-thread simulator validates resilience under sensor failures:

| Time window | Fault | Behavior |
|-------------|-------|----------|
| 0-10 s | Normal | σ converges from 1.87 m to 1.46 m |
| 10-15 s | GPS lost | σ rises to 3.94 m (IMU dead reckoning); vision still anchors |
| 15-18 s | GPS recovers | σ drops back to 1.20 m |
| 18-23 s | Vision low confidence | σ stays at 1.2 m (held by GPS); visual updates skipped |
| 23-30 s | Vision recovers | Mode D engagement resumes |

The closed-loop system tolerates single-modality failures of up to 5 s without
losing position lock.
