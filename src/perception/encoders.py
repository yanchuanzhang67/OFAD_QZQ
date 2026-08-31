"""Reusable sensor encoders for the lightweight ORAD BEV stack."""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ImageEncoder", "ImuEncoder"]


class ImageEncoder(nn.Module):
    """Compact image backbone with categorical depth prediction."""

    def __init__(self, in_channels: int, feature_channels: int,
                 depth_bins: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.depth_head = nn.Conv2d(feature_channels, depth_bins, 1)

    def forward(self, images):
        feature = self.backbone(images)
        depth = F.softmax(self.depth_head(feature), dim=1)
        return feature, depth


class ImuEncoder(nn.Module):
    """Encode a fixed-length IMU history into a BEV conditioning vector."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_channels, batch_first=True)
        self.proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, imu):
        _, hidden = self.gru(imu)
        return self.proj(hidden.squeeze(0))
