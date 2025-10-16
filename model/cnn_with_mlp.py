import torch.nn as nn

class MammographyClassifier(nn.Module):
    """CNN Backbone + 分類器"""
    
    def __init__(self, backbone, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        feature_dim = backbone.feature_dim
        
        # 分類器：將 4 張影像的特徵 concatenate
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images):
        # 提取特徵
        features = self.backbone(images)  # (B, 4, feature_dim)
        
        # Flatten 並分類
        features_flat = features.flatten(1)  # (B, 4 * feature_dim)
        logits = self.classifier(features_flat)
        
        return logits
