#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train + Test + Plot for MaMVT (stage-3 mid fusion, six-class)
+ ✅ TensorBoard logging support
"""
import os, json, random, argparse
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter  # ✅ NEW
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset import MammoDataset
from model.mamvt import MaMVT, MaMVTLoss
from model.maxvit import MaMVT_MaxViT
from model.backbone_4_channel import MaMVT_Shared 

VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")
CLASSES = 6
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_balanced_sampler(dataset, num_classes=CLASSES):
    labels = dataset.df["label"].to_numpy(dtype=np.int64)
    class_counts = np.bincount(labels, minlength=num_classes)
    print(f"[DATA] Class distribution: {class_counts}")

    class_weights = (1.0 / np.clip(class_counts, 1, None))
    class_weights = class_weights * (num_classes / class_weights.sum())
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True
    )
    return sampler


def evaluate_breast(model, loader, device, criterion=None, split="VAL", out_dir: Path = None):
    model.eval()
    all_y, all_predL, all_predR, all_predE = [], [], [], []
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{split}]", leave=False):
            for v in VIEWS:
                batch[v] = batch[v].to(device)
            y = batch["label"].to(device)
            out = model(batch)

            if criterion is not None:
                loss, _ = criterion(out, y)
                val_loss += loss.item()

            predL = out["left_logits"].argmax(dim=1)
            predR = out["right_logits"].argmax(dim=1)
            predE = out["exam_logits"].argmax(dim=1)

            all_y.append(y.cpu().numpy())
            all_predL.append(predL.cpu().numpy())
            all_predR.append(predR.cpu().numpy())
            all_predE.append(predE.cpu().numpy())

    y_true = np.concatenate(all_y)
    yL, yR, yE = np.concatenate(all_predL), np.concatenate(all_predR), np.concatenate(all_predE)

    repE = classification_report(y_true, yE, digits=4, zero_division=0)
    print(f"\n[{split}] Exam-level Report:\n{repE}")

    accE = accuracy_score(y_true, yE)
    f1E = f1_score(y_true, yE, average="macro", zero_division=0)
    val_loss = val_loss / len(loader) if criterion is not None else 0.0

    return {"acc": accE, "f1_macro": f1E, "loss": val_loss}, (y_true, yL, yR, yE)


def plot_confusion_matrix(cm, class_names, out_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def final_test(model, loader, device, out_dir: Path):
    print("\n[FINAL TEST]")
    metrics, (y_true, yL, yR, yE) = evaluate_breast(model, loader, device, split="TEST", out_dir=out_dir)
    class_names = [str(i) for i in range(CLASSES)]
    plot_confusion_matrix(confusion_matrix(y_true, yE), class_names, out_dir / "test_exam_confusion_matrix.png", "Exam-level CM")
    print(f"[PLOT] Saved confusion matrix -> {out_dir}")


def train(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))  # ✅ 初始化 TensorBoard writer

    train_ds = MammoDataset(args.train_csv, args.root_dir, args.img_size, train=True, in_chans=args.in_chans)
    val_ds   = MammoDataset(args.val_csv, args.root_dir, args.img_size, train=False, in_chans=args.in_chans)
    test_ds  = MammoDataset(args.test_csv, args.root_dir, args.img_size, train=False, in_chans=args.in_chans) if args.test_csv else None

    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
                             num_workers=args.workers, pin_memory=True) if test_ds else None

    if args.structure == "maxvit":
        model = MaMVT_MaxViT(
            backbone=args.backbone,
            pretrained=True,
            num_classes=CLASSES,
            in_chans=args.in_chans
        ).to(device)
    elif args.structure == "mamvt":
        model = MaMVT(backbone=args.backbone, pretrained=True, num_classes=CLASSES,
                    in_chans=args.in_chans).to(device)
    else:
        model = MaMVT_Shared(backbone=args.backbone, pretrained=True, num_classes=CLASSES,
                              in_chans=args.in_chans).to(device)

    criterion = MaMVTLoss(w_view=1.0, w_breast=1.0, w_exam=1.0,
                          class_weights=None, use_focal=(args.loss == "focal"),
                          label_smoothing=args.label_smoothing)
    print(f"[TRAIN] Using exam-level loss (w_exam=1.0)")

    if args.optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = GradScaler(enabled=not args.no_amp)
    best_f1, best_path = -1, out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_sampler = get_balanced_sampler(train_ds, CLASSES)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=False)

        model.train()
        pbar = tqdm(train_loader, desc=f"[TRAIN] {epoch}/{args.epochs}")
        run_loss = 0.0
        all_train_y, all_train_predE = [], []

        for i, batch in enumerate(pbar):
            for v in VIEWS:
                batch[v] = batch[v].to(device)
            y = batch["label"].to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=not args.no_amp):
                out = model(batch)
                loss, stats = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            run_loss += stats["loss"]

            predE = out["exam_logits"].argmax(dim=1)
            all_train_y.append(y.cpu().numpy())
            all_train_predE.append(predE.cpu().numpy())

        train_loss = run_loss / len(train_loader)
        train_acc = accuracy_score(np.concatenate(all_train_y), np.concatenate(all_train_predE))
        val_metrics, _ = evaluate_breast(model, val_loader, device, criterion=criterion, split="VAL", out_dir=out_dir)
        val_acc, val_f1, val_loss = val_metrics["acc"], val_metrics["f1_macro"], val_metrics["loss"]

        # ✅ TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1_macro/val", val_f1, epoch)
        for k in ["loss_view", "loss_breast", "loss_exam"]:
            if k in stats:
                writer.add_scalar(f"Loss/{k}", stats[k], epoch)

        log = {"epoch": epoch, "train_loss": train_loss, "metrics": val_metrics}
        (out_dir / "log.jsonl").open("a").write(json.dumps(log) + "\n")

        print(f"[EPOCH {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"[CKPT] Best model saved (f1_macro={best_f1:.4f})")

    writer.close()  # ✅ 關閉 TensorBoard writer

    if test_loader:
        model.load_state_dict(torch.load(best_path, map_location=device))
        final_test(model, test_loader, device, out_dir)

# ===============================================================
# CLI
# ===============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", default="")
    ap.add_argument("--root_dir", default=".")
    ap.add_argument("--out_dir", default="runs/mamvt_stage3_exam")
    ap.add_argument("--structure", type=str, default="mamvt", choices=["mamvt", "maxvit", "4channel"])
    ap.add_argument("--backbone", default="swin_base_patch4_window12_384_in22k")
    ap.add_argument("--img_size", type=int, default=1536)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--loss", type=str, default="focal", choices=["focal", "ce"])
    ap.add_argument("--label_smoothing", type=float, default=0.2)
    ap.add_argument("--in_chans", type=int, default=1, choices=[1, 3])
    ap.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adamw"])
    args = ap.parse_args()
    train(args)