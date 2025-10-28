# model/mamvt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm: pip install timm") from e

VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")

# ==========================================================
# Cross-Attention on 2D feature maps (B,C,H,W)
# ==========================================================
class CrossAttention2D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(proj_drop)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_q.shape
        q = x_q.flatten(2).transpose(1, 2)
        kv = x_kv.flatten(2).transpose(1, 2)
        q = self.norm_q(q); kv = self.norm_kv(kv)
        q = self.q_proj(q); k = self.k_proj(kv); v = self.v_proj(kv)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1); attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)
        return out.transpose(1, 2).view(B, C, H, W)

# ==========================================================
# Cross-View Fusion
# ==========================================================
class CrossViewFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, proj_drop: float = 0.0):
        super().__init__()
        self.ipsi_attn = CrossAttention2D(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.ipsi_attn_rev = CrossAttention2D(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.contra_attn = CrossAttention2D(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.contra_attn_rev = CrossAttention2D(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.norm = nn.BatchNorm2d(dim)

    def _bidir(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_upd = self.ipsi_attn(a, b); b_upd = self.ipsi_attn_rev(b, a)
        return a + a_upd, b + b_upd

    def _bidir_contra(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_upd = self.contra_attn(a, b); b_upd = self.contra_attn_rev(b, a)
        return a + a_upd, b + b_upd

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lcc, lmlo = feats["L-CC"], feats["L-MLO"]
        rcc, rmlo = feats["R-CC"], feats["R-MLO"]
        # 同側
        lcc, lmlo = self._bidir(lcc, lmlo)
        rcc, rmlo = self._bidir(rcc, rmlo)
        # 對側
        lcc, rcc = self._bidir_contra(lcc, rcc)
        lmlo, rmlo = self._bidir_contra(lmlo, rmlo)
        return {
            "L-CC": self.norm(lcc),
            "R-CC": self.norm(rcc),
            "L-MLO": self.norm(lmlo),
            "R-MLO": self.norm(rmlo),
        }

# ==========================================================
# Swin wrapper with stage3 mid fusion
# ==========================================================
class SwinMidFusionWrapper(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, fusion: CrossViewFusion,
                 in_chans: int = 1, drop_path_rate: float = 0.1):
        super().__init__()
        base = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=384,          # keep in sync with backbone variant
            drop_path_rate=drop_path_rate,
            num_classes=0,
            features_only=False
        )
        assert hasattr(base, 'layers'), 'Expected Swin-like backbone.'
        self.backbone = base
        self.fusion = fusion
        self.stage3 = self.backbone.layers[2]
        self.stage4 = self.backbone.layers[3]
        self.half_s3 = len(self.stage3.blocks) // 2
        self.out_channels = getattr(self.stage4, "out_dim", self.stage4.dim * 2)

    @staticmethod
    def _tokens_to_bchw(x, H, W):
        if x.ndim == 3:
            B, L, C = x.shape; x = x.view(B, H, W, C)
        elif x.ndim == 4:
            B, H, W, C = x.shape
        return x.permute(0, 3, 1, 2).contiguous()

    def _run_blocks(self, layer, x, H, W, start, end):
        for i in range(start, end):
            x = layer.blocks[i](x)
        return x

    def forward_single_until_mid(self, x):
        x = self.backbone.patch_embed(x)
        H, W = self.backbone.patch_embed.grid_size
        x = x.view(x.size(0), H, W, -1)
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x)
        x = self.backbone.layers[0](x)
        x = self.backbone.layers[1](x)
        if self.stage3.downsample is not None:
            x = self.stage3.downsample(x); H //= 2; W //= 2
        x = self._run_blocks(self.stage3, x, H, W, 0, self.half_s3)
        return x, H, W

    def forward_single_finish(self, x, H, W):
        x = self._run_blocks(self.stage3, x, H, W, self.half_s3, len(self.stage3.blocks))
        x = self.stage4(x); H //= 2; W //= 2
        x = self.backbone.norm(x)
        return self._tokens_to_bchw(x, H, W)

    def forward_fused(self, views: Dict[str, torch.Tensor]):
        tokens = {}; H = W = None
        for k, img in views.items():
            t, Hk, Wk = self.forward_single_until_mid(img)
            tokens[k] = t; H, W = Hk, Wk
        maps = {k: self._tokens_to_bchw(t, H, W) for k, t in tokens.items()}
        maps = self.fusion(maps)
        feats = {}
        for k, m in maps.items():
            x = m.permute(0, 2, 3, 1).contiguous()
            feats[k] = self.forward_single_finish(x, H, W)
        return feats

# ==========================================================
# MaMVT Model (with exam head)
# ==========================================================
class MaMVT(nn.Module):
    def __init__(self, backbone: str = "swin_base_patch4_window12_384_in22k",
                 pretrained: bool = True, num_classes: int = 6,
                 attn_heads: int = 8, proj_drop: float = 0.0,
                 in_chans: int = 1, drop_path_rate: float = 0.1):
        super().__init__()
        # Probe stage-3 dim for fusion
        tmp = timm.create_model(backbone, pretrained=pretrained, in_chans=in_chans, num_classes=0, img_size=384)
        s3 = tmp.layers[2]
        dim_s3 = getattr(s3, "out_dim", s3.dim * 2 if getattr(s3, "downsample", None) else s3.dim)
        del tmp

        fusion = CrossViewFusion(dim=dim_s3, num_heads=attn_heads, proj_drop=proj_drop)
        self.swin = SwinMidFusionWrapper(backbone, pretrained, fusion,
                                         in_chans=in_chans, drop_path_rate=drop_path_rate)

        feat_channels = self.swin.out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)

        # heads
        self.view_head = nn.Linear(feat_channels, num_classes)
        self.breast_left_head  = nn.Linear(feat_channels * 2, num_classes)
        self.breast_right_head = nn.Linear(feat_channels * 2, num_classes)
        self.exam_head = nn.Linear(num_classes * 2, num_classes)  # ✅ new: learnable exam fusion

        for m in [self.view_head, self.breast_left_head, self.breast_right_head, self.exam_head]:
            nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]):
        feats = self.swin.forward_fused({v: batch[v] for v in VIEWS})

        # pooled per-view
        pooled, view_logits = {}, {}
        for v in VIEWS:
            p = self.pool(feats[v]).flatten(1)
            pooled[v] = p
            view_logits[v] = self.view_head(p)

        # breast-level
        left_feat  = torch.cat([pooled["L-CC"],  pooled["L-MLO"]],  dim=1)
        right_feat = torch.cat([pooled["R-CC"], pooled["R-MLO"]], dim=1)
        left_logits  = self.breast_left_head(left_feat)
        right_logits = self.breast_right_head(right_feat)

        # ✅ exam-level fusion (learnable)
        exam_feat   = torch.cat([left_logits, right_logits], dim=1)
        exam_logits = self.exam_head(exam_feat)

        return {
            "view_logits": view_logits,
            "left_logits": left_logits,
            "right_logits": right_logits,
            "exam_logits": exam_logits,
        }

# ==========================================================
# Loss Function (Focal or CE with smoothing)
# ==========================================================
def focal_ce(logits, target, alpha=0.25, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits, target, weight=weight, reduction='none')
    pt = torch.exp(-ce)
    focal = alpha * (1 - pt) ** gamma * ce
    return focal.mean()

class MaMVTLoss(nn.Module):
    def __init__(self, w_view: float = 1.0, w_breast: float = 1.0, w_exam: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None,
                 use_focal: bool = True, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.2):
        super().__init__()
        self.w_view = w_view
        self.w_breast = w_breast
        self.w_exam = w_exam
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.alpha = alpha; self.gamma = gamma
        self.label_smoothing = label_smoothing

    def _ce_or_focal(self, logits, target):
        if self.use_focal:
            return focal_ce(logits, target, alpha=self.alpha, gamma=self.gamma, weight=self.class_weights)
        else:
            return F.cross_entropy(logits, target, weight=self.class_weights,
                                   label_smoothing=self.label_smoothing)

    def forward(self, outputs: Dict[str, Dict[str, torch.Tensor]], label: torch.Tensor):
        loss_view = sum(self._ce_or_focal(outputs["view_logits"][v], label) for v in VIEWS)
        loss_breast = self._ce_or_focal(outputs["left_logits"], label) + self._ce_or_focal(outputs["right_logits"], label)
        loss_exam = self._ce_or_focal(outputs["exam_logits"], label)  # ✅ new
        total = self.w_view * loss_view + self.w_breast * loss_breast + self.w_exam * loss_exam
        stats = {
            "loss": float(total.item()),
            "loss_view": float(loss_view.item()),
            "loss_breast": float(loss_breast.item()),
            "loss_exam": float(loss_exam.item()),
        }
        return total, stats

def build_mamvt(backbone: str = "swin_base_patch4_window12_384_in22k",
                pretrained: bool = True, num_classes: int = 6,
                attn_heads: int = 8, proj_drop: float = 0.0,
                in_chans: int = 1, drop_path_rate: float = 0.1) -> MaMVT:
    return MaMVT(backbone=backbone, pretrained=pretrained,
                 num_classes=num_classes, attn_heads=attn_heads,
                 proj_drop=proj_drop, in_chans=in_chans,
                 drop_path_rate=drop_path_rate)