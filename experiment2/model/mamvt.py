import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm: pip install timm") from e

VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")

# =============================================
# Cross-Attention on 2D feature maps (B,C,H,W)
# =============================================
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
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out

class CrossViewFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, proj_drop: float = 0.0):
        super().__init__()
        self.ipsi_attn = CrossAttention2D(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.ipsi_attn_rev = CrossAttention2D(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.contra_attn = CrossAttention2D(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.contra_attn_rev = CrossAttention2D(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.norm = nn.BatchNorm2d(dim)

    def _bidir(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_upd = self.ipsi_attn(a, b)
        b_upd = self.ipsi_attn_rev(b, a)
        return a + a_upd, b + b_upd

    def _bidir_contra(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_upd = self.contra_attn(a, b)
        b_upd = self.contra_attn_rev(b, a)
        return a + a_upd, b + b_upd

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lcc, lmlo = feats["L-CC"], feats["L-MLO"]
        rcc, rmlo = feats["R-CC"], feats["R-MLO"]
        lcc, lmlo = self._bidir(lcc, lmlo)
        rcc, rmlo = self._bidir(rcc, rmlo)
        lcc, rcc = self._bidir_contra(lcc, rcc)
        lmlo, rmlo = self._bidir_contra(lmlo, rmlo)
        return {
            "L-CC": self.norm(lcc),
            "R-CC": self.norm(rcc),
            "L-MLO": self.norm(lmlo),
            "R-MLO": self.norm(rmlo),
        }

# ---------------------------------
# Swin wrapper with mid-stage3 fusion (paper version, fixed for timm>=1.0)
# ---------------------------------
class SwinMidFusionWrapper(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, fusion: CrossViewFusion):
        super().__init__()
        base = timm.create_model(backbone_name, pretrained=pretrained)
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
            B, L, C = x.shape
            x = x.view(B, H, W, C)
        elif x.ndim == 4:
            B, H, W, C = x.shape
        else:
            raise ValueError(f"Unexpected x.ndim={x.ndim}")
        return x.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]

    @staticmethod
    def _bchw_to_tokens(x):
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

    def _run_blocks(self, layer, x, H, W, start, end):
        # timm >= 1.0 的 Swin block 期望輸入為 [B, H, W, C]
        for i in range(start, end):
            x = layer.blocks[i](x)
        return x

    def forward_single_until_mid(self, x):
        # patch embed → [B, H*W, C] → reshape to [B,H,W,C]
        x = self.backbone.patch_embed(x)
        H, W = self.backbone.patch_embed.grid_size
        x = x.view(x.size(0), H, W, -1)
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x)

        # stage-1, stage-2
        x = self.backbone.layers[0](x)
        x = self.backbone.layers[1](x)

        # ✅ 先做 stage3.downsample (新版 timm 只需傳 x)
        if self.stage3.downsample is not None:
            x = self.stage3.downsample(x)
            H //= 2
            W //= 2

        # stage-3 前半段
        x = self._run_blocks(self.stage3, x, H, W, 0, self.half_s3)
        return x, H, W

    def forward_single_finish(self, x, H, W):
        # 跑 stage-3 後半段 blocks（此時已經 downsample 過，別再 downsample）
        x = self._run_blocks(self.stage3, x, H, W, self.half_s3, len(self.stage3.blocks))

        # 直接進入 stage-4（timm>=1.0 直接傳 x）
        x = self.stage4(x)
        H //= 2; W //= 2

        x = self.backbone.norm(x)
        return self._tokens_to_bchw(x, H, W)

    def forward_fused(self, views: Dict[str, torch.Tensor]):
        tokens = {}
        H = W = None
        for k, img in views.items():
            t, Hk, Wk = self.forward_single_until_mid(img)
            tokens[k] = t
            H, W = Hk, Wk
        # 先轉 [B,H,W,C] 給 Cross-View Fusion
        maps = {k: self._tokens_to_bchw(t, H, W) for k, t in tokens.items()}
        maps = self.fusion(maps)
        # ✅ 不要再轉回 tokens，直接給 Swin blocks（因為新版 Swin 吃 [B,H,W,C]）
        feats = {}
        for k, m in maps.items():
            # m 現在是 [B,C,H,W]，轉為 [B,H,W,C]
            x = m.permute(0, 2, 3, 1).contiguous()
            feats[k] = self.forward_single_finish(x, H, W)
        return feats

# ---------------------------------
# MaMVT (fusion at stage3 mid, paper-aligned)
# ---------------------------------
class MaMVT(nn.Module):
    def __init__(self, backbone: str = "swin_base_patch4_window12_384", pretrained: bool = True,
                 num_classes: int = 6, attn_heads: int = 8, proj_drop: float = 0.0):
        super().__init__()
        tmp = timm.create_model(backbone, pretrained=pretrained)
        s3 = tmp.layers[2]
        # ✅ timm>=1.0：stage3.dim 是 downsample 之前；真正 block 的 C 是 out_dim
        dim_s3 = getattr(s3, "out_dim", s3.dim * 2 if getattr(s3, "downsample", None) is not None else s3.dim)
        del tmp

        fusion = CrossViewFusion(dim=dim_s3, num_heads=attn_heads, proj_drop=proj_drop)
        self.swin = SwinMidFusionWrapper(backbone, pretrained, fusion)
        feat_channels = self.swin.out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.view_head = nn.Linear(feat_channels, num_classes)
        self.breast_left_head = nn.Linear(feat_channels * 2, num_classes)
        self.breast_right_head = nn.Linear(feat_channels * 2, num_classes)
        for m in [self.view_head, self.breast_left_head, self.breast_right_head]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]):
        feats = self.swin.forward_fused({v: batch[v] for v in VIEWS})
        pooled, view_logits = {}, {}
        for v in VIEWS:
            p = self.pool(feats[v]).flatten(1)
            pooled[v] = p
            view_logits[v] = self.view_head(p)
        left_feat = torch.cat([pooled["L-CC"], pooled["L-MLO"]], dim=1)
        right_feat = torch.cat([pooled["R-CC"], pooled["R-MLO"]], dim=1)
        left_logits = self.breast_left_head(left_feat)
        right_logits = self.breast_right_head(right_feat)
        return {"view_logits": view_logits, "left_logits": left_logits, "right_logits": right_logits}

class MaMVTLoss(nn.Module):
    def __init__(self, w_view: float = 1.0, w_breast: float = 1.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.w_view = w_view
        self.w_breast = w_breast
        self.class_weights = class_weights.float() if class_weights is not None else None

    def forward(self, outputs: Dict[str, Dict[str, torch.Tensor]], label: torch.Tensor):
        loss_view = sum(F.cross_entropy(outputs["view_logits"][v], label, weight=self.class_weights) for v in VIEWS)
        loss_breast = F.cross_entropy(outputs["left_logits"], label, weight=self.class_weights) + F.cross_entropy(outputs["right_logits"], label, weight=self.class_weights)
        total = self.w_view * loss_view + self.w_breast * loss_breast
        stats = {"loss": float(total.item()), "loss_view": float(loss_view.item()), "loss_breast": float(loss_breast.item())}
        return total, stats

def build_mamvt(backbone: str = "swin_base_patch4_window12_384", pretrained: bool = True, num_classes: int = 6, attn_heads: int = 8, proj_drop: float = 0.0) -> MaMVT:
    return MaMVT(backbone=backbone, pretrained=pretrained, num_classes=num_classes, attn_heads=attn_heads, proj_drop=proj_drop)
