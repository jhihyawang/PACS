import torch.nn as nn
import torch

# class MammographyClassifier(nn.Module):
#     """CNN Backbone + 分類器"""
    
#     def __init__(self, backbone, num_classes: int = 2, dropout: float = 0.3):
#         super().__init__()
#         self.backbone = backbone
#         feature_dim = backbone.feature_dim
        
#         # 分類器：將 4 張影像的特徵 concatenate
#         self.classifier = nn.Sequential(
#             nn.Linear(feature_dim * 4, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
            
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
            
#             nn.Linear(512, num_classes)
#         )
    
#     def forward(self, images):
#         # 提取特徵
#         features = self.backbone(images)  # (B, 4, feature_dim)
        
#         # Flatten 並分類
#         features_flat = features.flatten(1)  # (B, 4 * feature_dim)
#         logits = self.classifier(features_flat)
        
#         return logits

class TransformerFusion(nn.Module):
    def __init__(self, in_dim, d_model=512, nhead=8, depth=2, dropout=0.1,
                 use_rel_tokens=False, pooling="cls"):
        super().__init__()
        self.use_rel_tokens = use_rel_tokens
        self.pooling = pooling  # "cls" or "attn"

        # 把 backbone 的 D 映到 Transformer 的 d
        self.proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()

        # 視角/左右嵌入（最多 8 個 token：4 個view + 4 個關係）
        max_tokens = 8 if use_rel_tokens else 4
        self.view_embed = nn.Embedding(max_tokens, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=depth)

        # attention pooling（若不用 CLS）
        self.att_pool = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats):  # feats: (B, 4, D)
        B, V, Din = feats.shape  # V=4
        x = self.proj(feats)     # (B, 4, d)

        # （可選）加入 4 個「關係 token」：左右差 & 同側視角差
        if self.use_rel_tokens:
            rel = torch.stack([
                x[:,2]-x[:,0],  # R-CC - L-CC
                x[:,3]-x[:,1],  # R-MLO - L-MLO
                x[:,1]-x[:,0],  # L-MLO - L-CC
                x[:,3]-x[:,2],  # R-MLO - R-CC
            ], dim=1)          # (B,4,d)
            x = torch.cat([x, rel], dim=1)  # (B,8,d)
            V = 8

        # 加 view/side embedding
        idx = torch.arange(V, device=x.device)
        x = x + self.view_embed(idx)[None, :, :]

        # 前置 [CLS]
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,d)
        seq = torch.cat([cls, x], dim=1)        # (B, V+1, d)

        h = self.tr(seq)                        # (B, V+1, d)

        if self.pooling == "cls":
            fused = h[:, 0]                     # (B, d)
        else:
            w = torch.softmax(self.att_pool(h[:, 1:]).squeeze(-1), dim=1)  # (B,V)
            fused = (h[:, 1:] * w.unsqueeze(-1)).sum(dim=1)                 # (B,d)

        return self.norm(fused)                 # (B, d)

class MammographyClassifier(nn.Module):
    def __init__(self, backbone, num_classes, dropout=0.25,
                 d_model=512, nhead=8, depth=2, use_rel_tokens=True):
        super().__init__()
        self.backbone = backbone                # 會回傳 (B,4,D)
        D = backbone.feature_dim                # e.g., 2048 or 1536

        self.fusion = TransformerFusion(
            in_dim=D, d_model=d_model, nhead=nhead, depth=depth,
            dropout=dropout, use_rel_tokens=use_rel_tokens, pooling="cls"
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, images):                  # images: (B,4,3,H,W)
        feats = self.backbone(images)           # (B,4,D)
        fused = self.fusion(feats)              # (B,d_model)
        logits = self.head(fused)               # (B,C)
        return logits