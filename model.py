"""Models retained for the four-view BI-RADS triage experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from cross_view_attention import SideInvariantCrossViewFusion


class SiameseResNetRuleModel(nn.Module):
    """Shared ConvNeXt encoder with baseline or four-view fusion heads.

    Input view order is L-CC, R-CC, L-MLO, R-MLO. The class name is retained to
    keep existing experiment scripts and checkpoint loading code compatible.
    """

    def __init__(
        self,
        backbone_name="convnext_tiny",
        pretrained=True,
        num_classes=3,
        architecture="baseline",
        concate_method="2fc",
        decision_rule="avg",
        use_asymmetry_auxiliary=False,
    ):
        super().__init__()
        if backbone_name != "convnext_tiny":
            raise ValueError("The retained experiments support only convnext_tiny")
        if architecture not in {"baseline", "cross_view"}:
            raise ValueError(f"Unsupported architecture: {architecture}")
        if concate_method not in {"concat", "2fc"}:
            raise ValueError(f"Unsupported concatenation method: {concate_method}")
        if decision_rule not in {"avg", "conditional"}:
            raise ValueError(f"Unsupported decision rule: {decision_rule}")

        self.backbone_name = backbone_name
        self.architecture = architecture
        self.num_classes = num_classes
        self.concate_method = concate_method
        self.decision_rule = decision_rule
        self.use_asymmetry_auxiliary = bool(use_asymmetry_auxiliary)
        self.feature_dim = 768

        self.backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
        self.backbone.classifier[2] = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.sideinv_fusion = (
            SideInvariantCrossViewFusion(
                dim=self.feature_dim,
                heads=4,
                mode="cross_view",
            )
            if architecture == "cross_view"
            else None
        )

        if concate_method == "2fc":
            self.breast_classifier = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, self.num_classes),
            )
        else:
            self.breast_classifier = nn.Linear(self.feature_dim * 2, self.num_classes)

        if decision_rule == "conditional":
            self.axis_view_classifier = nn.Linear(self.feature_dim, 2)
            self.axis_view_attention = nn.Linear(self.feature_dim, 2, bias=False)
            self.axis_side_attention = nn.Linear(self.feature_dim * 2, 2, bias=False)
            nn.init.zeros_(self.axis_view_attention.weight)
            nn.init.zeros_(self.axis_side_attention.weight)
        else:
            self.axis_view_classifier = None
            self.axis_view_attention = None
            self.axis_side_attention = None

        self.aux_asymmetry_classifier = (
            nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, 1),
            )
            if self.use_asymmetry_auxiliary
            else None
        )
        self.last_conditional_logits = None
        self.last_aux_asymmetry_logit = None

    def forward_one_view(self, images):
        return self.backbone.features(images)

    def _fuse_views(self, feature_maps):
        if self.sideinv_fusion is None:
            return feature_maps
        named = {
            "L-CC": feature_maps[:, 0],
            "R-CC": feature_maps[:, 1],
            "L-MLO": feature_maps[:, 2],
            "R-MLO": feature_maps[:, 3],
        }
        fused = self.sideinv_fusion(named)
        return torch.stack(
            [fused["L-CC"], fused["R-CC"], fused["L-MLO"], fused["R-MLO"]],
            dim=1,
        )

    def _conditional_probabilities(self, pooled, left_features, right_features):
        # Reorder from input layout to left CC/MLO followed by right CC/MLO.
        views = [pooled[:, 0], pooled[:, 2], pooled[:, 1], pooled[:, 3]]
        view_logits = [self.axis_view_classifier(view) for view in views]

        left_weights = F.softmax(
            torch.stack([self.axis_view_attention(views[0]), self.axis_view_attention(views[1])], dim=1),
            dim=1,
        )
        right_weights = F.softmax(
            torch.stack([self.axis_view_attention(views[2]), self.axis_view_attention(views[3])], dim=1),
            dim=1,
        )
        left_logits = (left_weights * torch.stack(view_logits[:2], dim=1)).sum(dim=1)
        right_logits = (right_weights * torch.stack(view_logits[2:], dim=1)).sum(dim=1)

        side_weights = F.softmax(
            torch.stack(
                [self.axis_side_attention(left_features), self.axis_side_attention(right_features)],
                dim=1,
            ),
            dim=1,
        )
        axis_logits = (
            side_weights * torch.stack([left_logits, right_logits], dim=1)
        ).sum(dim=1)
        self.last_conditional_logits = axis_logits

        p_action = torch.sigmoid(axis_logits[:, 0])
        p_type = torch.sigmoid(axis_logits[:, 1])
        return torch.stack(
            [p_action * (1.0 - p_type), 1.0 - p_action, p_action * p_type],
            dim=1,
        )

    def forward(self, images):
        if images.ndim != 5 or images.shape[1] != 4:
            raise ValueError(f"Expected (B, 4, C, H, W), got {tuple(images.shape)}")
        batch, views, channels, height, width = images.shape
        maps = self.forward_one_view(images.reshape(batch * views, channels, height, width))
        maps = maps.reshape(batch, views, *maps.shape[1:])
        maps = self._fuse_views(maps)

        pooled = self.global_pool(maps.flatten(0, 1)).flatten(1)
        pooled = pooled.reshape(batch, views, self.feature_dim)
        left_features = torch.cat([pooled[:, 0], pooled[:, 2]], dim=1)
        right_features = torch.cat([pooled[:, 1], pooled[:, 3]], dim=1)

        self.last_aux_asymmetry_logit = (
            self.aux_asymmetry_classifier(
                torch.abs(left_features - right_features)
            ).squeeze(1)
            if self.aux_asymmetry_classifier is not None
            else None
        )

        left_logits = self.breast_classifier(left_features)
        right_logits = self.breast_classifier(right_features)
        left_probabilities = F.softmax(left_logits, dim=1)
        right_probabilities = F.softmax(right_logits, dim=1)

        if self.decision_rule == "conditional":
            exam_probabilities = self._conditional_probabilities(
                pooled, left_features, right_features
            )
        else:
            self.last_conditional_logits = None
            exam_probabilities = (left_probabilities + right_probabilities) / 2.0

        exam_log_probabilities = torch.log(exam_probabilities.clamp_min(1e-8))
        return (
            exam_log_probabilities,
            left_probabilities,
            right_probabilities,
            left_logits,
            right_logits,
        )
