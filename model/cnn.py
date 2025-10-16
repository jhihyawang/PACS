import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, EfficientNet_B3_Weights
from typing import Literal


class MammographyCNNBackbone(nn.Module):
    """
    乳房影像分析的 CNN Backbone
    - 共用權重處理 4 張影像 (L-CC, L-MLO, R-CC, R-MLO)
    - 輸出每張影像的 global feature vector
    """
    
    def __init__(
        self,
        backbone: Literal['resnet50', 'efficientnet_b0', 'efficientnet_b3'] = 'resnet50',
        feature_dim: int = 2048,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone: 選擇的 backbone 架構
            feature_dim: 輸出 feature 的維度
            pretrained: 是否使用 ImageNet 預訓練權重
            freeze_backbone: 是否凍結 backbone 權重
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # ===== 建立 CNN backbone =====
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
            # 移除最後的 FC layer 和 avgpool
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out_channels = 2048
            
        elif backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            efficientnet = models.efficientnet_b0(weights=weights)
            self.backbone = efficientnet.features
            backbone_out_channels = 1280
            
        elif backbone == 'efficientnet_b3':
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            efficientnet = models.efficientnet_b3(weights=weights)
            self.backbone = efficientnet.features
            backbone_out_channels = 1536
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 凍結 backbone 權重（可選）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✓ Backbone weights frozen")
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection（將 backbone 輸出投影到指定維度）
        if backbone_out_channels != feature_dim:
            self.feature_proj = nn.Sequential(
                nn.Linear(backbone_out_channels, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.feature_proj = nn.Identity()
        
        self.backbone_out_channels = backbone_out_channels
    
    def forward_single_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        處理單張影像
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, feature_dim)
        """
        # CNN backbone 提取特徵
        features = self.backbone(x)  # (B, C, H', W')
        
        # Global Average Pooling
        features = self.gap(features)  # (B, C, 1, 1)
        features = features.flatten(1)  # (B, C)
        
        # Feature projection
        features = self.feature_proj(features)  # (B, feature_dim)
        
        return features
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        處理 4 張影像（共用權重）
        Args:
            images: (B, 4, 3, H, W) - 4 張影像 (L-CC, L-MLO, R-CC, R-MLO)
        Returns:
            features: (B, 4, feature_dim) - 每張影像的 global feature
        """
        B, num_views, C, H, W = images.shape
        assert num_views == 4, "Expected 4 views (L-CC, L-MLO, R-CC, R-MLO)"
        
        # 將 4 張影像展平成 batch
        images_flat = images.view(B * num_views, C, H, W)  # (B*4, 3, H, W)
        
        # 共用 backbone 處理所有影像
        features_flat = self.forward_single_image(images_flat)  # (B*4, feature_dim)
        
        # 重新組織成 (B, 4, feature_dim)
        features = features_flat.view(B, num_views, self.feature_dim)
        
        return features
    
    def get_num_params(self) -> dict:
        """計算模型參數量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
    
    def unfreeze_backbone(self):
        """解凍 backbone 權重"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"✓ Backbone weights unfrozen")
    
    def freeze_backbone(self):
        """凍結 backbone 權重"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"✓ Backbone weights frozen")


# ============= 使用範例 =============

if __name__ == "__main__":
    # 檢查 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\n")
    
    # ===== 建立模型 =====
    model = MammographyCNNBackbone(
        backbone='resnet50',           # 'resnet50', 'efficientnet_b0', 'efficientnet_b3'
        feature_dim=2048,              # 輸出 feature 維度
        pretrained=True,               # 使用 ImageNet 預訓練權重
        freeze_backbone=False          # 是否凍結 backbone
    ).to(device)
    
    print("="*70)
    print(f"Model: {model.backbone_name.upper()}")
    print("="*70)
    
    # 模型資訊
    params = model.get_num_params()
    print(f"Parameters:")
    print(f"  Total:     {params['total']:>12,}")
    print(f"  Trainable: {params['trainable']:>12,}")
    print(f"  Frozen:    {params['frozen']:>12,}")
    
    # 估算參數記憶體
    param_memory_gb = params['total'] * 4 / (1024**3)
    print(f"\nParameter Memory: {param_memory_gb:.2f} GB")
    
    # ===== 測試前向傳播 =====
    print("\n" + "="*70)
    print("Testing Forward Pass")
    print("="*70)
    
    # 建議的影像大小
    img_size = 512  # 可改為 768
    batch_size = 4
    
    # 建立 dummy 資料
    dummy_images = torch.randn(batch_size, 4, 3, img_size, img_size).to(device)
    print(f"Input shape:  {dummy_images.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num views:  4 (L-CC, L-MLO, R-CC, R-MLO)")
    print(f"  - Image size: {img_size}x{img_size}")
    
    # 前向傳播
    model.eval()
    with torch.no_grad():
        features = model(dummy_images)
    
    print(f"\nOutput shape: {features.shape}")
    print(f"  - Each sample has 4 feature vectors")
    print(f"  - Each feature vector is {model.feature_dim}-dimensional")
    
    # ===== 記憶體估算 =====
    print("\n" + "="*70)
    print("Memory Estimation (RTX 3090 - 24GB)")
    print("="*70)
    
    def estimate_memory(bs, img_sz, model_name):
        input_mem = bs * 4 * 3 * img_sz * img_sz * 4 / (1024**3)
        
        if model_name == 'resnet50':
            param_mem = 25e6 * 4 / (1024**3)
            act_mem = input_mem * 12
        elif model_name == 'efficientnet_b0':
            param_mem = 5e6 * 4 / (1024**3)
            act_mem = input_mem * 8
        else:  # efficientnet_b3
            param_mem = 12e6 * 4 / (1024**3)
            act_mem = input_mem * 10
        
        total = input_mem + param_mem + act_mem
        return {
            'input': input_mem,
            'param': param_mem,
            'activation': act_mem,
            'total': total
        }
    
    configs = [
        (2, 512, 'resnet50'),
        (4, 512, 'resnet50'),
        (8, 512, 'resnet50'),
        (2, 768, 'resnet50'),
        (4, 768, 'resnet50'),
    ]
    
    print(f"\n{'Batch':>5} | {'Size':>7} | {'Backbone':>15} | {'Total Memory':>12} | {'Status':>6}")
    print("-" * 70)
    
    for bs, img_sz, backbone in configs:
        mem = estimate_memory(bs, img_sz, backbone)
        status = "✓" if mem['total'] < 20 else "✗"
        print(f"{bs:>5} | {img_sz:>7} | {backbone:>15} | {mem['total']:>10.2f} GB | {status:>6}")
    
    # ===== 建議配置 =====
    print("\n" + "="*70)
    print("Recommended Configuration")
    print("="*70)
    print("1. Image size: 512x512, Batch size: 4-8")
    print("2. Image size: 768x768, Batch size: 2-4")
    print("3. Use mixed precision training (torch.cuda.amp) to save memory")
    print("4. Use gradient accumulation if need larger effective batch size")
    print("\nBackbone Options:")
    print("  - ResNet50:        Strong features, ~25M params")
    print("  - EfficientNet-B0: Memory efficient, ~5M params")
    print("  - EfficientNet-B3: Balanced, ~12M params")
    
    # ===== 儲存與載入範例 =====
    print("\n" + "="*70)
    print("Save & Load Example")
    print("="*70)
    
    print("""
# 儲存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'backbone': 'resnet50',
    'feature_dim': 2048,
}, 'mammography_backbone.pth')

# 載入模型
checkpoint = torch.load('mammography_backbone.pth')
model = MammographyCNNBackbone(
    backbone=checkpoint['backbone'],
    feature_dim=checkpoint['feature_dim'],
    pretrained=False
)
model.load_state_dict(checkpoint['model_state_dict'])
    """)