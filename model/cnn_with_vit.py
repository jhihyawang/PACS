import torch.nn as nn
import torch

class MammographyViT(nn.Module):
    """CNN Backbone + Vision Transformer"""
    
    def __init__(
        self,
        backbone,
        num_classes: int = 2,
        vit_dim: int = 768,
        vit_depth: int = 4,
        vit_heads: int = 8,
        vit_mlp_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.backbone = backbone
        feature_dim = backbone.feature_dim
        
        # Feature projection to ViT dimension
        self.to_vit = nn.Linear(feature_dim, vit_dim)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_dim))
        
        # Positional embedding (4 views + 1 CLS token = 5)
        self.pos_embedding = nn.Parameter(torch.randn(1, 5, vit_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=vit_heads,
            dim_feedforward=vit_mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=vit_depth)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, num_classes)
        )
    
    def forward(self, images):
        # 提取 CNN features
        features = self.backbone(images)  # (B, 4, feature_dim)
        
        B = features.shape[0]
        
        # Project to ViT dimension
        x = self.to_vit(features)  # (B, 4, vit_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, vit_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 5, vit_dim)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)  # (B, 5, vit_dim)
        
        # Classification using CLS token
        cls_output = x[:, 0]  # (B, vit_dim)
        logits = self.classifier(cls_output)  # (B, num_classes)
        
        return logits