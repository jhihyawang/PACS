#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train + Test + Plot for MaMVT (stage-3 mid fusion, six-class, no patch head)
- 每個 epoch 印出左右乳房 classification report
- 訓練後自動在 test set 上評估、輸出混淆矩陣與報告（含 breast-level 與 exam-level）
- 產生訓練曲線圖 (loss/acc/f1)
"""
import os, math, json, random, argparse
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset import MammoDataset
from model.mamvt import MaMVT, MaMVTLoss

VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")
CLASSES = 6
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def exam_logits(outputs:Dict[str,torch.Tensor]):
    return (outputs['left_logits']+outputs['right_logits'])/2.

def evaluate_breast(model, loader, device, split="VAL", out_dir:Path=None):
    model.eval(); all_y, all_predL, all_predR = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{split}]", leave=False):
            for v in VIEWS: batch[v]=batch[v].to(device)
            y=batch['label'].to(device)
            out=model(batch)
            predL=torch.argmax(out['left_logits'],1)
            predR=torch.argmax(out['right_logits'],1)
            all_y.append(y.cpu().numpy()); all_predL.append(predL.cpu().numpy()); all_predR.append(predR.cpu().numpy())
    y_true=np.concatenate(all_y); yL=np.concatenate(all_predL); yR=np.concatenate(all_predR)
    repL=classification_report(y_true,yL,digits=4)
    repR=classification_report(y_true,yR,digits=4)
    print(f"\n[{split}] Left Breast Report:\n{repL}\n[{split}] Right Breast Report:\n{repR}")
    if out_dir:
        (out_dir/f"{split.lower()}_left_report.txt").write_text(repL)
        (out_dir/f"{split.lower()}_right_report.txt").write_text(repR)
    acc=(accuracy_score(y_true,yL)+accuracy_score(y_true,yR))/2
    f1m=(f1_score(y_true,yL,average='macro')+f1_score(y_true,yR,average='macro'))/2
    return {"acc":acc,"f1_macro":f1m}, (y_true, yL, yR)

def plot_training_curves(log_file:Path, out_dir:Path):
    if not log_file.exists():
        print("[WARN] No log.jsonl for plotting."); return
    epochs, acc, f1m, loss = [], [], [], []
    for line in log_file.open():
        obj=json.loads(line); epochs.append(obj['epoch']); acc.append(obj['metrics']['acc']); f1m.append(obj['metrics']['f1_macro']); loss.append(obj['train_loss'])
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label='Train Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(epochs,acc,label='Val Acc'); plt.plot(epochs,f1m,label='Val F1_macro'); plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.grid(True)
    plt.suptitle('Training Curves')
    plt.tight_layout(); plt.savefig(out_dir/'training_curves.png'); plt.close()
    print(f"[PLOT] Saved training curves -> {out_dir/'training_curves.png'}")

def plot_confusion_matrix(cm,class_names,out_path:Path,title:str):
    plt.figure(figsize=(8,6)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()

def final_test(model,loader,device,out_dir:Path):
    print("\n[FINAL TEST]"); metrics,(y_true,yL,yR)=evaluate_breast(model,loader,device,split="TEST",out_dir=out_dir)
    # Breast-level confusion
    cmL=confusion_matrix(y_true,yL); cmR=confusion_matrix(y_true,yR); class_names=[str(i) for i in range(CLASSES)]
    plot_confusion_matrix(cmL,class_names,out_dir/'test_left_confusion_matrix.png','Left Breast Confusion Matrix')
    plot_confusion_matrix(cmR,class_names,out_dir/'test_right_confusion_matrix.png','Right Breast Confusion Matrix')
    # Exam-level average
    model.eval(); all_y, all_pred_exam = [], []
    with torch.no_grad():
        for batch in loader:
            for v in VIEWS: batch[v]=batch[v].to(device)
            y=batch['label'].to(device)
            out=model(batch)
            pred_exam=torch.argmax((out['left_logits']+out['right_logits'])/2,1)
            all_y.append(y.cpu().numpy()); all_pred_exam.append(pred_exam.cpu().numpy())
    y_true=np.concatenate(all_y); y_exam=np.concatenate(all_pred_exam)
    report_exam=classification_report(y_true,y_exam,digits=4)
    print(f"\n[TEST] Exam-level Classification Report:\n{report_exam}")
    cm_exam=confusion_matrix(y_true,y_exam)
    plot_confusion_matrix(cm_exam,class_names,out_dir/'test_exam_confusion_matrix.png','Exam-level Confusion Matrix')
    print(f"[PLOT] Saved exam-level confusion matrix -> {out_dir/'test_exam_confusion_matrix.png'}")

def train(args):
    set_seed(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    train_ds=MammoDataset(args.train_csv,args.root_dir,args.img_size,train=True)
    val_ds=MammoDataset(args.val_csv,args.root_dir,args.img_size,train=False)
    test_ds=MammoDataset(args.test_csv,args.root_dir,args.img_size,train=False) if args.test_csv else None
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True)
    val_loader=DataLoader(val_ds,batch_size=max(1,args.batch_size//2),shuffle=False,num_workers=args.workers,pin_memory=True)
    test_loader=DataLoader(test_ds,batch_size=max(1,args.batch_size//2),shuffle=False,num_workers=args.workers,pin_memory=True) if test_ds else None

    model=MaMVT(backbone=args.backbone,pretrained=True,num_classes=CLASSES).to(device)
    class_weights=None
    if args.class_weights and Path(args.class_weights).exists():
        w=torch.tensor(np.load(args.class_weights),dtype=torch.float32); class_weights=w.to(device)
    criterion=MaMVTLoss(w_view=1.0,w_breast=1.0,class_weights=class_weights)
    opt=optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler=GradScaler(enabled=not args.no_amp)
    best_f1=-1; best_path=out_dir/'best.pt'

    for epoch in range(1,args.epochs+1):
        model.train(); pbar=tqdm(train_loader,desc=f"[TRAIN] {epoch}/{args.epochs}"); run_loss=0.
        for i,batch in enumerate(pbar):
            for v in VIEWS: batch[v]=batch[v].to(device)
            y=batch['label'].to(device); opt.zero_grad(set_to_none=True)
            with autocast(enabled=not args.no_amp):
                out=model(batch); loss,stats=criterion(out,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss+=stats['loss']; pbar.set_postfix({"loss":f"{run_loss/(i+1):.4f}"})
        val_metrics,_=evaluate_breast(model,val_loader,device,split="VAL",out_dir=out_dir)
        log={"epoch":epoch,"train_loss":run_loss/len(train_loader),"metrics":val_metrics}
        (out_dir/'log.jsonl').open('a').write(json.dumps(log)+'\n')
        if val_metrics['f1_macro']>best_f1:
            best_f1=val_metrics['f1_macro']; torch.save(model.state_dict(),best_path)
            print(f"[CKPT] best model saved (f1_macro={best_f1:.4f})")

    plot_training_curves(out_dir/'log.jsonl',out_dir)
    if test_loader:
        model.load_state_dict(torch.load(best_path,map_location=device))
        final_test(model,test_loader,device,out_dir)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_csv',required=True); ap.add_argument('--val_csv',required=True); ap.add_argument('--test_csv',default='')
    ap.add_argument('--root_dir',default='.'); ap.add_argument('--out_dir',default='runs/mamvt_stage3')
    ap.add_argument('--backbone',default='swin_base_patch4_window12_384'); ap.add_argument('--img_size',type=int,default=1536)
    ap.add_argument('--batch_size',type=int,default=4); ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--weight_decay',type=float,default=1e-4)
    ap.add_argument('--workers',type=int,default=6); ap.add_argument('--no_amp',action='store_true'); ap.add_argument('--class_weights',default='')
    args=ap.parse_args(); train(args)