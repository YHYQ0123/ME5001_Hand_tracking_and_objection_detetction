# === stm_yolo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C2f

class YOLOCls_BackBone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.backbone = nn.Sequential(
            Conv(in_channels, 16, k=3, s=2),           # P1 / 2
            Conv(16, 32, k=3, s=2),                    # P2 / 4
            C2f(32, 32, n=1, shortcut=True),
            Conv(32, 64, k=3, s=2),                    # P3 / 8
            C2f(64, 64, n=2, shortcut=True),
            Conv(64, 128, k=3, s=2),                   # P4 / 16
            C2f(128, 128, n=2, shortcut=True),
            Conv(128, 256, k=3, s=2),                  # P5 / 32
            C2f(256, 256, n=1, shortcut=True)
        )

    def forward(self, x):
        return self.backbone(x)

class MultiViewVideoClassifier(nn.Module):
    def __init__(self, num_classes=2, sequence_length=5):
        super().__init__()
        self.common_backbone = YOLOCls_BackBone(in_channels=3)
        self.kinect_backbone = YOLOCls_BackBone(in_channels=3)
        self.feature_dim_per_frame = 256

        self.temporal_aggregator = nn.Sequential(
            nn.Conv1d(self.feature_dim_per_frame * 3, self.feature_dim_per_frame * 3 // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.feature_dim_per_frame * 3 // 2, self.feature_dim_per_frame * 3 // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.aggregated_feature_dim = self.feature_dim_per_frame * 3 // 4

        self.fusion_head = nn.Sequential(
            nn.Linear(self.aggregated_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, view1, view2, view3):
        B, T, C, H, W = view1.shape
        combined_left_right_flat = torch.cat((view1, view2), dim=0).view(B * T * 2, C, H, W)
        kinect_flat = view3.view(B * T, C, H, W)

        combined_left_right_feats = self.common_backbone(combined_left_right_flat)
        kinect_feats = self.kinect_backbone(kinect_flat)

        combined_left_right_feats_vec = F.adaptive_avg_pool2d(combined_left_right_feats, (1, 1)).squeeze(-1).squeeze(-1)
        kinect_feats_vec = F.adaptive_avg_pool2d(kinect_feats, (1, 1)).squeeze(-1).squeeze(-1)

        feats_view1_vec = combined_left_right_feats_vec[:B*T]
        feats_view2_vec = combined_left_right_feats_vec[B*T:]

        all_views_flat_combined = torch.cat((feats_view1_vec, feats_view2_vec, kinect_feats_vec), dim=1)
        combined_temporal_input = all_views_flat_combined.view(B, T, self.feature_dim_per_frame * 3)
        final_fused_features = self.temporal_aggregator(combined_temporal_input.permute(0, 2, 1)).squeeze(-1)
        logits = self.fusion_head(final_fused_features)
        return logits
