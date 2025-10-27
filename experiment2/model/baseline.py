from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


VIEWS = ["L-CC", "R-CC", "L-MLO", "R-MLO"]

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x).flatten(1)


class MultiViewMammoNet(nn.Module):
    def __init__(self, num_classes: int = 6, shared_backbone: bool = False, backbone_name: str = "resnet18"):
        super().__init__()
        self.shared_backbone = shared_backbone
        feat_dim = 512 if backbone_name in ("resnet18","resnet34") else 2048

        def make_backbone():
            if backbone_name == "resnet18":
                m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            elif backbone_name == "resnet34":
                m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            layers = list(m.children())[:-2] # drop avgpool+fc → [B,C,H,W]
            return nn.Sequential(*layers)

        if shared_backbone:
            self.backbone = make_backbone()
        else:
            self.backbone = nn.ModuleDict({v: make_backbone() for v in VIEWS})

        self.gap = GlobalAvgPool()

        # 2× views per side → feature dim doubles
        self.left_head = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes)
        )
        self.right_head = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes)
        )

    def encode_one(self, x, view: str):
        if self.shared_backbone:
            feat = self.backbone(x)
        else:
            feat = self.backbone[view](x)
        return self.gap(feat) # [B, C]


    def forward(self, batch: Dict[str, torch.Tensor]):
        # Expect keys: L-CC, L-MLO, R-MLO, R-CC
        f = {v: self.encode_one(batch[v], v) for v in VIEWS}
        left = torch.cat([f["L-CC"], f["L-MLO"]], dim=1)
        right = torch.cat([f["R-MLO"], f["R-CC"]], dim=1)
        left_logits = self.left_head(left)
        right_logits = self.right_head(right)
        left_prob = F.softmax(left_logits, dim=1)
        right_prob = F.softmax(right_logits, dim=1)
        final_prob = (left_prob + right_prob) / 2.0
        return {"left": left_prob, "right": right_prob, "final": final_prob, "left_logits": left_logits, "right_logits": right_logits}