import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C2f, Classify

class YOLOCls_BackBone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.backbone = nn.Sequential(
            Conv(in_channels, 16, k=3, s=2),
            Conv(16, 32, k=3, s=2),
            C2f(32, 32, n=1, shortcut=True),
            Conv(32, 64, k=3, s=2),
            C2f(64, 64, n=2, shortcut=True),
            Conv(64, 128, k=3, s=2),
            C2f(128, 128, n=2, shortcut=True),
            Conv(128, 256, k=3, s=2),
            C2f(256, 256, n=1, shortcut=True)
        )

    def forward(self, x):
        return self.backbone(x)

class YOLOBaselineClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = YOLOCls_BackBone(in_channels=3)
        self.head = Classify(256, num_classes)  # 256 是最后一层输出通道数

    def forward(self, view1, view2, view3):
        last_frame = view3[:, -1, :, :, :]  # 仅保留 view3 的最后一帧: [B, C, H, W]
        features = self.backbone(last_frame)
        output = self.head(features)
        return output
