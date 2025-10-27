#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train + Test MaMVT (stage-3 mid fusion, six-class, no patch head)
During training: shows breast-level classification report (left/right).
After training: directly evaluates on test set if provided.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data.dataset_mamvt import MammoDataset
from model_mamvt import MaMVT, MaMVTLoss

VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")
CLASSES = 6
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def build_loaders(train_csv, val_csv, test_csv, root_dir, img_size, batch_size, workers):
    train_ds = MammoDataset(train_csv, root_dir, img_size, train=True)
    val_ds   = MammoDataset(val_csv, root_dir, img_size, train=False)
    test_ds  = MammoDataset(test_csv, root_dir, img_size, train=False) if test_csv else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1,batch_size//2), shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=max(1,batch_size//2), shuffle=False, num_workers=workers, pin_memory=True) if test_csv else None
    return {"train":train_loader,"val":val_loader,"test":test_loader}

def exam_logits(outputs:Dict[str,torch.Tensor]):
    return (outputs['left_logits']+outputs['right_logits'])/2.

def evaluate(model, loader, device, split="VAL"):
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
    acc=(accuracy_score(y_true,yL)+accuracy_score(y_true,yR))/2
    f1m=(f1_score(y_true,yL,average='macro')+f1_score(y_true,yR,average='macro'))/2
    return {"acc":acc,"f1_macro":f1m}

def train(args):
    set_seed(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir=Path(args.out_dir); save_dir.mkdir(parents=True,exist_ok=True)
    loaders=build_loaders(args.train_csv,args.val_csv,args.test_csv,args.root_dir,args.img_size,args.batch_size,args.workers)
    model=MaMVT(backbone=args.backbone,pretrained=True,num_classes=CLASSES).to(device)
    weights=None
    if args.class_weights and Path(args.class_weights).exists():
        w=torch.tensor(np.load(args.class_weights),dtype=torch.float32); weights=w.to(device)
    criterion=MaMVTLoss(w_view=1.0,w_breast=1.0,class_weights=weights)
    opt=optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler=GradScaler(enabled=not args.no_amp)

    best_f1=-1; best_path=save_dir/"best.pt"
    for epoch in range(1,args.epochs+1):
        model.train(); pbar=tqdm(loaders['train'],desc=f"[TRAIN] {epoch}/{args.epochs}"); tot=0
        for i,batch in enumerate(pbar):
            for v in VIEWS: batch[v]=batch[v].to(device)
            y=batch['label'].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=not args.no_amp):
                out=model(batch); loss,stats=criterion(out,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot+=stats['loss']; pbar.set_postfix({"loss":f"{tot/(i+1):.4f}"})
        metrics=evaluate(model,loaders['val'],device,split="VAL")
        print(f"[EPOCH {epoch}] val acc={metrics['acc']:.4f} f1_macro={metrics['f1_macro']:.4f}")
        if metrics['f1_macro']>best_f1:
            best_f1=metrics['f1_macro']; torch.save(model.state_dict(),best_path)
            print(f"[CKPT] best model saved (f1_macro={best_f1:.4f})")

    # Final evaluation on test if provided
    if loaders['test']:
        print("\n[FINAL TEST]"); model.load_state_dict(torch.load(best_path, map_location=device))
        evaluate(model,loaders['test'],device,split="TEST")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_csv',required=True); ap.add_argument('--val_csv',required=True)
    ap.add_argument('--test_csv',default=''); ap.add_argument('--root_dir',default='.')
    ap.add_argument('--out_dir',default='runs/mamvt_stage3')
    ap.add_argument('--backbone',default='swin_base_patch4_window12_384')
    ap.add_argument('--img_size',type=int,default=1536); ap.add_argument('--batch_size',type=int,default=4)
    ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--weight_decay',type=float,default=1e-4); ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--no_amp',action='store_true'); ap.add_argument('--class_weights',default='')
    args=ap.parse_args(); train(args)
