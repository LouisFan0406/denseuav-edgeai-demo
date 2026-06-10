"""Pose Refinement Head with shared ResNet18 backbone (ImageNet pretrained).

This is the inference-ready model definition for the partial 3D pose head used
in the UAV self-localization pipeline.

Design (matches the final report):
    UAV (3ch, 224x224)  -> ResNet18 (shared, pretrained) -> 512-d UAV feature
    Sat (3ch, 224x224)  -> ResNet18 (shared, pretrained) -> 512-d Sat feature

    fused = concat(UAV, Sat, |UAV - Sat|)   (1536-d)
    fused -> MLP (1536 -> 512 -> 256)
                 |
                 +-- head_xy  (mu_x, mu_y, log_var_x, log_var_y)
                 +-- head_yaw (mu_yaw, log_var_yaw)
                 +-- head_z   (4 altitude logits: 40 / 50 / 60 / 80 m)

ImageNet normalization is applied internally. The dataset only needs to provide
tensors in [0, 1] range.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class PoseRefinementHeadResNet18(nn.Module):
    """Shared ResNet18 Siamese pose refinement head."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_altitude_classes: int = 4,
        pretrained: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_altitude_classes = num_altitude_classes

        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        self.feat_dim = 512

        fuse_in = self.feat_dim * 3
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.head_xy = nn.Linear(latent_dim, 4)
        self.head_yaw = nn.Linear(latent_dim, 2)
        self.head_z = nn.Linear(latent_dim, num_altitude_classes)

    def _normalize(self, img_01: torch.Tensor) -> torch.Tensor:
        return (img_01 - self.img_mean) / self.img_std

    def encode(self, img_01: torch.Tensor) -> torch.Tensor:
        x = self._normalize(img_01)
        feat = self.backbone(x)
        return feat.flatten(1)

    def forward(self, uav: torch.Tensor, sat: torch.Tensor) -> dict[str, torch.Tensor]:
        uav_feat = self.encode(uav)
        sat_feat = self.encode(sat)
        diff = (uav_feat - sat_feat).abs()

        fused = torch.cat([uav_feat, sat_feat, diff], dim=1)
        latent = self.fuse(fused)

        xy = self.head_xy(latent)
        yaw = self.head_yaw(latent)
        z_logits = self.head_z(latent)

        return {
            "mu_xy": xy[:, :2],
            "log_var_xy": xy[:, 2:],
            "mu_yaw": yaw[:, 0],
            "log_var_yaw": yaw[:, 1],
            "z_logits": z_logits,
            "latent": latent,
        }
