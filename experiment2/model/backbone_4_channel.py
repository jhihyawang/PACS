import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple

# 四個視角的固定順序
VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")


# ==========================================================
# Cross-Attention 模組（與原 MaMVT 相同）
# ==========================================================
class CrossAttention2D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, proj_drop: float = 0.0):
        super().__init__()
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
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)
        return out.transpose(1, 2).view(B, C, H, W)


# ==========================================================
# Cross-View Fusion 模組（與原 MaMVT 相同）
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


# ==========================================================
# 單 Backbone + 四視角並行 MaMVT
# ==========================================================
class MaMVT_Shared(nn.Module):
    def __init__(
        self,
        backbone: str = "maxvit_small_tf_384",
        pretrained: bool = True,
        num_classes: int = 6,
        in_chans: int = 1,
        img_size: int = 384,
        attn_heads: int = 8,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        # 建立單一 backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes)
        self.fusion = None

        # 用 dummy forward 推測 feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            feat = self.backbone(dummy)
            if isinstance(feat, torch.Tensor):
                dim = feat.shape[1]
            else:
                dim = feat[-1].shape[1]
        print(f"[DEBUG] Shared backbone feature dim = {dim}")

        self.fusion = CrossViewFusion(dim=dim, num_heads=attn_heads, proj_drop=proj_drop)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 視圖、左右乳與整體 exam classifier
        self.view_head = nn.Linear(dim, num_classes)
        self.breast_head = nn.Linear(dim * 2, num_classes)
        self.exam_head = nn.Linear(num_classes * 2, num_classes)

        for m in [self.view_head, self.breast_head, self.exam_head]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, 4, C, H, W)
        """
        B, V, C, H, W = x.shape
        assert V == 4, f"Expected 4 views, got {V}"

        # ---- (1) flatten 視角維度，單 backbone forward ----
        x = x.view(B * V, C, H, W)
        feats = self.backbone(x)  # (B*4, D, h, w)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]  # 若是tuple取最後一層
        D, h, w = feats.shape[1:]
        feats = feats.view(B, V, D, h, w)

        # ---- (2) reshape 成字典 ----
        feat_dict = {vname: feats[:, i] for i, vname in enumerate(VIEWS)}

        # ---- (3) cross-view fusion ----
        fused = self.fusion(feat_dict)

        # ---- (4) 視角 pooling 與分類 ----
        pooled = {k: self.pool(f).flatten(1) for k, f in fused.items()}
        view_logits = {k: self.view_head(pooled[k]) for k in VIEWS}

        # 左右乳融合
        left_feat = torch.cat([pooled["L-CC"], pooled["L-MLO"]], dim=1)
        right_feat = torch.cat([pooled["R-CC"], pooled["R-MLO"]], dim=1)
        left_logits = self.breast_head(left_feat)
        right_logits = self.breast_head(right_feat)

        # exam-level 融合
        exam_logits = torch.maximum(left_logits, right_logits)

        return {
            "view_logits": view_logits,
            "left_logits": left_logits,
            "right_logits": right_logits,
            "exam_logits": exam_logits,
        }
