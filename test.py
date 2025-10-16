# pip install torch torchvision pandas numpy tqdm scikit-learn matplotlib seaborn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# For evaluation metrics and visualization
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from model.cnn import MammographyCNNBackbone
from dataloader import MammographyDataset
from model.cnn_with_mlp import MammographyClassifier

class Tester:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        num_classes: int,
        class_names: List[str] = None,
        use_amp: bool = True
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
        self.use_amp = use_amp
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 結果儲存
        self.results = {
            'predictions': [],
            'probabilities': [],
            'true_labels': [],
            'sample_ids': [],
            'test_loss': 0.0,
            'test_acc': 0.0
        }
    
    def test(self) -> Dict:
        """執行測試"""
        print("="*70)
        print("Starting Model Testing")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Test batches: {len(self.test_loader)}")
        print("="*70)
        
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_probs = []
        all_labels = []
        all_sample_ids = []
        
        pbar = tqdm(self.test_loader, desc="Testing")
        
        with torch.no_grad():
            for batch in pbar:
                # 解包 batch
                if len(batch) == 3:
                    images, labels, meta = batch
                    sample_ids = meta.get('sample_id', ['unknown'] * len(labels))
                else:
                    images, labels = batch
                    sample_ids = ['unknown'] * len(labels)
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向傳播
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # 計算機率和預測
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # 統計
                running_loss += loss.item() * images.size(0)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 儲存結果
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_sample_ids.extend(sample_ids)
                
                # 更新進度條
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
        
        # 計算最終指標
        test_loss = running_loss / total
        test_acc = 100. * correct / total
        
        # 儲存結果
        self.results.update({
            'predictions': all_preds,
            'probabilities': all_probs,
            'true_labels': all_labels,
            'sample_ids': all_sample_ids,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
        
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Acc:  {test_acc:.2f}%")
        print("-"*70)
        
        return self.results
    
    def calculate_metrics(self) -> Dict:
        """計算詳細評估指標"""
        y_true = self.results['true_labels']
        y_pred = self.results['predictions']
        y_prob = np.array(self.results['probabilities'])
        
        # 基本指標
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # 平均指標
        precision_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[0]
        recall_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[1]
        f1_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2]
        
        precision_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[0]
        recall_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[1]
        f1_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[2]
        
        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        
        # AUC (如果是二分類或多分類)
        auc_scores = {}
        try:
            if self.num_classes == 2:
                auc_scores['binary'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc_scores['macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                auc_scores['weighted'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except ValueError as e:
            print(f"Warning: Could not calculate AUC - {e}")
        
        metrics = {
            'accuracy': accuracy,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'auc_scores': auc_scores,
            'class_names': self.class_names
        }
        
        return metrics
    
    def print_detailed_results(self, metrics: Dict):
        """打印詳細結果"""
        print("\n" + "="*70)
        print("DETAILED TEST RESULTS")
        print("="*70)
        
        print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Macro F1-Score:   {metrics['f1_macro']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        
        if metrics['auc_scores']:
            print(f"AUC Scores: {metrics['auc_scores']}")
        
        print("\nPer-Class Results:")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 70)
        
        for i, class_name in enumerate(self.class_names):
            precision = metrics['precision_per_class'][i]
            recall = metrics['recall_per_class'][i]
            f1 = metrics['f1_per_class'][i]
            support = metrics['support_per_class'][i]
            
            print(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
        
        print("-" * 70)
        print(f"{'Macro Avg':<15} {metrics['precision_macro']:<10.4f} {metrics['recall_macro']:<10.4f} {metrics['f1_macro']:<10.4f} {sum(metrics['support_per_class']):<10}")
        print(f"{'Weighted Avg':<15} {metrics['precision_weighted']:<10.4f} {metrics['recall_weighted']:<10.4f} {metrics['f1_weighted']:<10.4f} {sum(metrics['support_per_class']):<10}")
        
        print("\nConfusion Matrix:")
        print("-" * 40)
        cm = np.array(metrics['confusion_matrix'])
        
        # 打印混淆矩陣
        print("True\\Pred", end="")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name[:8]:>10}", end="")
        print()
        
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name[:8]:<10}", end="")
            for j in range(len(self.class_names)):
                print(f"{cm[i, j]:>10}", end="")
            print()
    
    def save_results(self, save_dir: str, prefix: str = "test"):
        """儲存測試結果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 計算指標
        metrics = self.calculate_metrics()
        
        # 儲存詳細結果
        results_to_save = {
            'test_loss': self.results['test_loss'],
            'test_acc': self.results['test_acc'],
            'metrics': metrics,
            'predictions': self.results['predictions'],
            'true_labels': self.results['true_labels'],
            'sample_ids': self.results['sample_ids']
        }
        
        # 儲存為 JSON
        with open(save_dir / f'{prefix}_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=4, default=str)
        
        # 儲存預測結果為 CSV
        predictions_df = pd.DataFrame({
            'sample_id': self.results['sample_ids'],
            'true_label': self.results['true_labels'],
            'predicted_label': self.results['predictions'],
            'correct': np.array(self.results['true_labels']) == np.array(self.results['predictions'])
        })
        
        # 添加機率
        probabilities = np.array(self.results['probabilities'])
        for i in range(self.num_classes):
            predictions_df[f'prob_class_{i}'] = probabilities[:, i]
        
        predictions_df.to_csv(save_dir / f'{prefix}_predictions.csv', index=False)
        
        print(f"\n✓ Results saved to:")
        print(f"  - {save_dir / f'{prefix}_results.json'}")
        print(f"  - {save_dir / f'{prefix}_predictions.csv'}")
        
        return metrics
    
    def plot_confusion_matrix(self, save_dir: str = None, prefix: str = "test"):
        """繪製混淆矩陣"""
        metrics = self.calculate_metrics()
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f'{prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to: {save_dir / f'{prefix}_confusion_matrix.png'}")
        
        plt.show()


# ===== 主測試函數 =====
def main(args):
    # 設定 device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # ===== 1. 建立測試 Dataset =====
    print("="*70)
    print("1. Loading Test Dataset")
    print("="*70)
    
    test_dataset = MammographyDataset(
        csv_path=args.test_csv,
        data_root=args.test_dir,
        img_size=args.img_size,
        is_train=False  # 測試時不使用數據增強
    )
    
    print(f"\nTest dataset size: {len(test_dataset)}")
    
    # 檢查類別分佈
    if hasattr(test_dataset, 'df'):
        label_counts = test_dataset.df['label'].value_counts().sort_index()
        print(f"Test label distribution:")
        for label, count in label_counts.items():
            print(f"  Class {label}: {count} samples")
    
    # ===== 2. 建立 DataLoader =====
    print("\n" + "="*70)
    print("2. Creating Test DataLoader")
    print("="*70)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 測試時不打亂
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # ===== 3. 載入模型 =====
    print("\n" + "="*70)
    print("3. Loading Model")
    print("="*70)
    
    # 建立模型架構
    backbone = MammographyCNNBackbone(
        backbone=args.backbone,
        feature_dim=args.feature_dim,
        pretrained=False  # 測試時不需要預訓練權重，會載入訓練好的權重
    )
    
    model = MammographyClassifier(
        backbone=backbone,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)
    
    # 載入訓練好的權重
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model weights loaded from checkpoint")
            if 'best_val_acc' in checkpoint:
                print(f"  Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        else:
            # 直接載入模型權重
            model.load_state_dict(checkpoint)
            print(f"✓ Model weights loaded")
    else:
        print("Error: No checkpoint specified!")
        return
    
    # 模型資訊
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.backbone.upper()}")
    print(f"Total parameters: {total_params:,}")
    
    # ===== 4. 執行測試 =====
    print("\n" + "="*70)
    print("4. Running Test")
    print("="*70)
    
    # 設定類別名稱
    class_names = [f'Category {i}' for i in range(args.num_classes)]
    
    tester = Tester(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=args.num_classes,
        class_names=class_names,
        use_amp=args.use_amp
    )
    
    # 執行測試
    results = tester.test()
    
    # 計算詳細指標
    metrics = tester.calculate_metrics()
    
    # 打印詳細結果
    tester.print_detailed_results(metrics)
    
    # ===== 5. 儲存結果 =====
    if args.save_results:
        print("\n" + "="*70)
        print("5. Saving Results")
        print("="*70)
        
        save_dir = args.results_dir
        tester.save_results(save_dir, prefix="test")
        
        # 繪製並儲存混淆矩陣
        if args.plot_cm:
            tester.plot_confusion_matrix(save_dir, prefix="test")
    
    print("\n" + "="*70)
    print("Testing Completed!")
    print("="*70)
    print(f"Final Test Accuracy: {results['test_acc']:.2f}%")
    print(f"Final Test Loss: {results['test_loss']:.4f}")
    print("="*70)


# ===== 命令列參數 =====
def parse_args():
    parser = argparse.ArgumentParser(description='Test Mammography Classifier')
    
    # 資料相關
    parser.add_argument('--test_csv', type=str, default='datasets/test_label.csv',
                        help='Test CSV path')
    parser.add_argument('--test_dir', type=str, default='datasets/test',
                        help='Test data directory')
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
    
    # 測試相關
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # 結果儲存
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save test results')
    parser.add_argument('--results_dir', type=str, default='./test_results',
                        help='Directory to save results')
    parser.add_argument('--plot_cm', action='store_true', default=True,
                        help='Plot and save confusion matrix')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
