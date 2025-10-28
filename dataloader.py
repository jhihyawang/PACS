# pip install torch torchvision pandas numpy pillow
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict

class MammographyDataset(Dataset):
    """
    簡化版資料集 - 不使用 Albumentations
    只使用 torchvision transforms
    """
    
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        img_size: int = 512,
        is_train: bool = True
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.is_train = is_train
        
        # 讀取 CSV
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} samples from {csv_path}")
        
        # 統計標籤
        label_counts = self.df['label'].value_counts().sort_index()
        print(f"Label distribution:")
        for label, count in label_counts.items():
            print(f"  Class {label}: {count} samples ({count/len(self.df)*100:.1f}%)")
        
        # 設定轉換
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        # 讀取 4 張影像
        view_names = ['L-CC', 'R-CC', 'L-MLO', 'R-MLO']
        images = []
        
        for view in view_names:
            img_path = self.data_root / row[view]
            
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))
            
            img = self.transform(img)
            images.append(img)
        
        # Stack 成 (4, 3, H, W)
        images = torch.stack(images, dim=0)
        label = int(row['label'])
        
        return images, label


# ===== 使用範例 =====
if __name__ == "__main__":
    
    # ===== 1. 建立 Dataset =====
    print("="*70)
    print("1. Create Dataset")
    print("="*70)
    
    # 訓練集
    train_dataset = MammographyDataset(
        csv_path='datasets/train_label.csv',
        data_root='datasets/train',
        img_size=512,
        is_train=True
    )
    
    # 驗證集
    val_dataset = MammographyDataset(
        csv_path='datasets/val_label.csv',
        data_root='datasets/val',
        img_size=512,
        is_train=False
    )
    
    # 測試集
    test_dataset = MammographyDataset(
        csv_path='datasets/test_label.csv',
        data_root='datasets/test',
        img_size=512,
        is_train=False
    )
    
    print(f"\nTotal samples:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    # ===== 2. 建立 DataLoader =====
    print("\n" + "="*70)
    print("2. Create DataLoader")
    print("="*70)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # ===== 3. 測試讀取資料 =====
    print("\n" + "="*70)
    print("3. Test Data Loading")
    print("="*70)
    
    # 讀取一個 batch
    images, labels, meta = next(iter(train_loader))
    
    print(f"Batch images shape: {images.shape}")  # (B, 4, 3, H, W)
    print(f"Batch labels shape: {labels.shape}")  # (B,)
    print(f"Labels in batch: {labels.tolist()}")
    
    print(f"\nFirst sample:")
    print(f"  Label: {labels[0].item()}")
    print(f"  Description (first 100 chars): {meta['description'][0][:100]}...")
    print(f"  Paths:")
    for view in ['L-CC', 'R-CC', 'L-MLO', 'R-MLO']:
        print(f"    {view}: {meta['paths'][view][0]}")
    
    # ===== 4. 檢查影像數值範圍 =====
    print("\n" + "="*70)
    print("4. Check Image Statistics")
    print("="*70)
    
    print(f"Min value: {images.min().item():.4f}")
    print(f"Max value: {images.max().item():.4f}")
    print(f"Mean value: {images.mean().item():.4f}")
    print(f"Std value: {images.std().item():.4f}")
