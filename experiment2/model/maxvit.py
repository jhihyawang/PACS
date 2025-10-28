import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import timm

VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")


# ==========================================================
# Cross-Attention (與 Swin 版相同)
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
# Cross-View Fusion (與 Swin 版相同)
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


# ==========================================================
# MaxViT Wrapper（替代 Swin Wrapper）
# ==========================================================
class MaxViTMidFusionWrapper(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, fusion: CrossViewFusion,
                 in_chans: int = 1, drop_path_rate: float = 0.1, img_size: int = 384):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=in_chans, num_classes=0, drop_path_rate=drop_path_rate
        )
        assert hasattr(self.backbone, 'stages'), 'Expected MaxViT-like model with stages.'
        self.stages = self.backbone.stages
        self.stem = self.backbone.stem
        self.fusion = fusion

        # 🔍 用 dummy forward 量測「經過最後一個 stage」後的通道數（不經 head）
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            x = self.stem(dummy)
            x = self.stages[0](x)
            x = self.stages[1](x)
            x = self.stages[2](x)
            x = self.stages[3](x)
            c_out = x.shape[1]
        self.out_channels = c_out
        print(f"[DEBUG] Wrapper out_channels = {self.out_channels}")

    def forward_fused(self, views: Dict[str, torch.Tensor]):
        maps = {}
        # 先跑到 fusion 接點
        for k, img in views.items():
            x = self.stem(img)
            x = self.stages[0](x)
            x = self.stages[1](x)
            x = self.stages[2](x)          # ← 在這個輸出做 cross-view fusion
            maps[k] = x

        # Cross-view fusion（B,C,H,W）
        fused = self.fusion(maps)

        # 再接續跑剩下的 stage 到輸出
        out_feats = {}
        for k, x in fused.items():
            x = self.stages[3](x)
            # ❌ 不經 self.backbone.head(x)；保留 2D 特徵
            out_feats[k] = x
        return out_feats

# ==========================================================
# MaMVT (MaxViT 版)
# ==========================================================
class MaMVT_MaxViT(nn.Module):
    def __init__(self, backbone: str = "maxvit_small_tf_384", pretrained: bool = True, num_classes: int = 6,
                 attn_heads: int = 8, proj_drop: float = 0.0, in_chans: int = 1, drop_path_rate: float = 0.1,
                 img_size: int = 384):
        super().__init__()
        # 建立一個暫時 backbone 來量測通道數
        tmp = timm.create_model(backbone, pretrained=pretrained, in_chans=in_chans, num_classes=0, drop_path_rate=drop_path_rate)
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            x = tmp.stem(dummy)
            x = tmp.stages[0](x)
            x = tmp.stages[1](x)
            x = tmp.stages[2](x)            # ← 我們的融合要接在這個輸出
            dim_s3 = x.shape[1]             # ← 直接量 C，最準
        print(f"[DEBUG] Using CrossAttention dim = {dim_s3}")

        fusion = CrossViewFusion(dim=dim_s3, num_heads=attn_heads, proj_drop=proj_drop)

        self.backbone_fused = MaxViTMidFusionWrapper(
            backbone, pretrained, fusion,
            in_chans=in_chans, drop_path_rate=drop_path_rate,
            img_size=img_size                # 傳給 wrapper 做輸出通道偵測
        )
        feat_channels = self.backbone_fused.out_channels

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.view_head = nn.Linear(feat_channels, num_classes)
        self.breast_left_head = nn.Linear(feat_channels * 2, num_classes)
        self.breast_right_head = nn.Linear(feat_channels * 2, num_classes)

        for m in [self.view_head, self.breast_left_head, self.breast_right_head]:
            nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]):
        feats = self.backbone_fused.forward_fused({v: batch[v] for v in VIEWS})
        pooled, view_logits = {}, {}

        # 每個視角單獨分類
        for v in VIEWS:
            p = self.pool(feats[v]).flatten(1)
            pooled[v] = p
            view_logits[v] = self.view_head(p)

        # 左右乳融合
        left_feat = torch.cat([pooled["L-CC"], pooled["L-MLO"]], dim=1)
        right_feat = torch.cat([pooled["R-CC"], pooled["R-MLO"]], dim=1)

        left_logits = self.breast_left_head(left_feat)
        right_logits = self.breast_right_head(right_feat)

        # ✅ 加上 exam-level 融合：取左右平均或最大
        exam_logits = torch.maximum(left_logits, right_logits)
        # 或者用更激進的策略：
        # exam_logits = torch.maximum(left_logits, right_logits)

        return {
            "view_logits": view_logits,
            "left_logits": left_logits,
            "right_logits": right_logits,
            "exam_logits": exam_logits,   # ✅ 新增這行
        }
