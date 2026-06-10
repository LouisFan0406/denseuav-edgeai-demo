# Pre-trained Model Checkpoint

The demo requires `pose_head_demo.pth`, a pre-trained pose refinement head
checkpoint (PyTorch state dict).

## Checkpoint Details

| Property | Value |
|----------|-------|
| Architecture | ResNet18 Siamese (shared backbone) |
| Parameters | 12 M |
| File size | ~46 MB |
| Trained on | JINMEI + XICHOU + DINO parks |
| Validated on | HUAZHONG park (zero-shot cross-area) |

## How to Obtain

The checkpoint is not committed to the public repository due to ongoing
research and possible patent considerations. To request access for academic
collaboration or verification:

- **Author:** 范詩涵 Shih-Han Fan
- **Email:** ledlab2392@gmail.com
- **Affiliation:** NTUST, EE5354701 Edge AI Final Project

Please include the following in your request:

1. Your name and affiliation
2. The purpose of your request (verification, collaboration, comparison, etc.)
3. Whether you need only the demo checkpoint, or also the training pipeline

## Placement

After obtaining the file, place it here:

```
checkpoints/pose_head_demo.pth
```

Then run:

```bash
python ../inference_demo.py
```

## Loading Format

The checkpoint is a PyTorch state dict, either:

- Raw `state_dict` saved via `torch.save(model.state_dict(), ...)`, OR
- A wrapper dict with key `"model_state_dict"` (as produced by `scripts/train_pose_head_resnet.py`)

The demo code handles both formats transparently.
