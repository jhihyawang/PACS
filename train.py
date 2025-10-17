# pip install torch torchvision pandas numpy tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from model.cnn import MammographyCNNBackbone
from dataloader import MammographyDataset
from model.cnn_with_mlp import MammographyClassifier
from model.cnn_with_vit import MammographyViT 


# ===== 訓練器 =====
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 損失函數 (支援類別權重)
        if config.get('class_weights') is not None:
            class_weights = torch.FloatTensor(config['class_weights']).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using class weights: {config['class_weights']}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 優化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 學習率排程器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
        
        # Mixed Precision Training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient Accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # 記錄
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """訓練一個 epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # 解包 batch (可能有 meta，也可能沒有)
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向傳播
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.accumulation_steps
                
                # 反向傳播
                self.scaler.scale(loss).backward()
                
                # Gradient Accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 統計
            running_loss += loss.item() * images.size(0) * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """驗證"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # 解包 batch
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 記錄預測結果
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """完整訓練流程"""
        print("="*70)
        print("Start Training")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation Steps: {self.accumulation_steps}")
        print(f"Effective Batch Size: {self.config['batch_size'] * self.accumulation_steps}")
        print("="*70)
        
        for epoch in range(self.config['epochs']):
            # 訓練
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 驗證
            val_loss, val_acc = self.validate(epoch)
            
            # 學習率排程
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 記錄
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # 打印統計
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # 儲存最佳模型 (基於 accuracy)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True, metric='acc')
                print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
            
            # 儲存最佳模型 (基於 loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True, metric='loss')
            
            # 定期儲存
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            print("-"*70)
        
        print("="*70)
        print("Training Completed!")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("="*70)
        
        # 儲存訓練歷史
        self.save_history()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, metric: str = 'acc'):
        """儲存檢查點"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            path = save_dir / f'best_model_{metric}.pth'
        # else:
        #     path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """儲存訓練歷史"""
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"✓ Training history saved to {save_dir / 'training_history.json'}")
    
    def load_checkpoint(self, path: str):
        """載入檢查點"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded from {path}")
        print(f"  Epoch: {checkpoint['epoch']+1}")
        print(f"  Best Val Acc: {self.best_val_acc:.2f}%")


# ===== 主訓練函數 =====
def main(args):
    # 設定隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 設定 device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # ===== 1. 建立 Dataset =====
    print("="*70)
    print("1. Loading Dataset")
    print("="*70)
    
    train_dataset = MammographyDataset(
        csv_path=args.train_csv,
        data_root=args.train_dir,
        img_size=args.img_size,
        is_train=True
    )
    
    val_dataset = MammographyDataset(
        csv_path=args.val_csv,
        data_root=args.val_dir,
        img_size=args.img_size,
        is_train=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    
    # ===== 2. 處理類別不平衡 (可選) =====
    class_weights = None
    sampler = None

    if args.use_class_weights:
        print("\n" + "="*70)
        print("2. Handling Class Imbalance")
        print("="*70)
        
        # 計算類別權重
        label_counts = train_dataset.df['label'].value_counts().to_dict()
        total_samples = len(train_dataset)
        num_classes = len(label_counts)
        
        class_weights = [total_samples / (num_classes * label_counts[i]) 
                        for i in sorted(label_counts.keys())]

        beta = 0.9999  # 超參數，控制權重強度
        
        effective_num = [1.0 - beta**label_counts[i] 
                        for i in sorted(label_counts.keys())]
        
        class_weights = [
            (1.0 - beta) / en 
            for en in effective_num
        ]
        
        # 歸一化
        sum_weights = sum(class_weights)
        class_weights = [w / sum_weights * num_classes for w in class_weights]

        print(f"Class counts: {label_counts}")
        print(f"Class weights: {class_weights}")
        
        if args.use_weighted_sampler:
            # 使用 WeightedRandomSampler
            sample_weights = []
            for idx in range(len(train_dataset)):
                label = train_dataset.df.iloc[idx]['label']
                weight = 1.0 / label_counts[label]
                sample_weights.append(weight)
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            print("Using WeightedRandomSampler")
        else:
            print("Using class weights in loss function")
    
    # ===== 3. 建立 DataLoader =====
    print("\n" + "="*70)
    print("3. Creating DataLoader")
    print("="*70)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),  # 如果使用 sampler 就不 shuffle
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    
    # ===== 4. 建立模型 =====
    print("\n" + "="*70)
    print("4. Building Model")
    print("="*70)
    
    backbone = MammographyCNNBackbone(
        backbone=args.backbone,
        feature_dim=args.feature_dim,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    )
    
    model = MammographyClassifier(
        backbone=backbone,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)
    
    # 模型資訊
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.backbone.upper()}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    
    # ===== 5. 訓練配置 =====
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'min_lr': args.min_lr,
        'use_amp': args.use_amp,
        'accumulation_steps': args.accumulation_steps,
        'class_weights': class_weights,
        'save_dir': args.save_dir,
        'save_freq': args.save_freq,
    }
    
    # ===== 6. 開始訓練 =====
    print("\n" + "="*70)
    print("5. Start Training")
    print("="*70)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # 如果有 checkpoint，載入
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 訓練
    trainer.train()
    
    print("\n✓ Training completed!")
    print(f"✓ Best model saved to: {args.save_dir}/best_model_acc.pth")


# ===== 命令列參數 =====
def parse_args():
    parser = argparse.ArgumentParser(description='Train Mammography Classifier')
    
    # 資料相關
    parser.add_argument('--train_csv', type=str, default='datasets/train_labels.csv',
                        help='Training CSV path')
    parser.add_argument('--train_dir', type=str, default='cropped_datasets',
                        help='Training data directory')
    parser.add_argument('--val_csv', type=str, default='datasets/val_labels.csv',
                        help='Validation CSV path')
    parser.add_argument('--val_dir', type=str, default='cropped_datasets',
                        help='Validation data directory')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Image size')
    
    # 模型相關
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0', 'efficientnet_b3'],
                        help='Backbone architecture')
    parser.add_argument('--feature_dim', type=int, default=2048,
                        help='Feature dimension')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights')
    
    # 訓練相關
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Minimum learning rate')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    
    # 類別不平衡處理
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights')
    parser.add_argument('--use_weighted_sampler', action='store_true',
                        help='Use weighted sampler')
    
    # 其他
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Save directory')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save frequency (epochs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)