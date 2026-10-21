#!/usr/bin/env python3
"""Unified OOF and fixed-test evaluation for retained experiments."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MammoDataset
from evaluation_utils import bootstrap_intervals, calculate_metrics, save_confusion_matrix
from model import SiameseResNetRuleModel


EXPERIMENTS = {
    "baseline": {"prefix": "baseline", "architecture": "baseline", "concat": "2fc", "rule": "avg", "aux": False},
    "cross_view": {"prefix": "cross_view", "architecture": "cross_view", "concat": "2fc", "rule": "avg", "aux": False},
    "baseline_aux": {"prefix": "baseline_aux", "architecture": "baseline", "concat": "2fc", "rule": "avg", "aux": True},
    "cross_view_aux": {"prefix": "cross_view_aux", "architecture": "cross_view", "concat": "2fc", "rule": "avg", "aux": True},
    "conditional": {"prefix": "conditional", "architecture": "baseline", "concat": "concat", "rule": "conditional", "aux": False},
    "conditional_aux": {"prefix": "conditional_aux", "architecture": "baseline", "concat": "concat", "rule": "conditional", "aux": True},
}
SEEDS = (42, 43, 44, 45, 46)
PROBABILITY_COLUMNS = ["p_birads0", "p_birads123", "p_birads45"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("oof", "test"):
        command = subparsers.add_parser(mode)
        command.add_argument("--experiment", choices=EXPERIMENTS, required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--root-dir", default="data/raw/datasets")
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--num-workers", type=int, default=4)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--bootstrap-samples", type=int, default=2000)
        command.add_argument("--bootstrap-seed", type=int, default=20261021)
    subparsers.choices["test"].add_argument("--csv", default="csv/three_classes/test_labels.csv")
    return parser


def find_checkpoint(experiment, run_id, fold):
    config = EXPERIMENTS[experiment]
    prefix = config["prefix"]
    matches = glob.glob(f"experiments/{prefix}_cv_fold{fold}_seed*_{run_id}")
    matches.extend(glob.glob(f"experiments/{prefix}_fold{fold}_seed*_{run_id}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {experiment} fold {fold} directory, found {matches}")
    checkpoint = Path(matches[0]) / "checkpoints/best_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def make_model(experiment, device):
    config = EXPERIMENTS[experiment]
    return SiameseResNetRuleModel(
        backbone_name="convnext_tiny",
        pretrained=False,
        num_classes=3,
        architecture=config["architecture"],
        concate_method=config["concat"],
        decision_rule=config["rule"],
        use_asymmetry_auxiliary=config["aux"],
    ).to(device)


def make_loader(csv_path, args, seed):
    dataset = MammoDataset(
        str(csv_path), args.root_dir, (1024, 512), train=False, seed=seed
    )
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device == "cuda",
    )


def predict(checkpoint, experiment, loader, device, description):
    model = make_model(experiment, device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    probabilities, labels = [], []
    with torch.inference_mode():
        for images, target in tqdm(loader, desc=description):
            probabilities.append(model(images.to(device, non_blocking=True))[0].exp().cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def summarize(labels, probabilities, args, title):
    metrics = calculate_metrics(labels, probabilities)
    metrics["bootstrap_95_ci"] = bootstrap_intervals(
        labels, probabilities, args.bootstrap_samples, args.bootstrap_seed
    )
    save_confusion_matrix(labels, probabilities, args.output_dir / "confusion_matrix.png", title)
    return metrics


def evaluate_oof(args, device):
    frames = []
    for fold, seed in enumerate(SEEDS):
        csv_path = Path(f"csv/e17_folds/fold_{fold}/val_labels.csv")
        source = pd.read_csv(csv_path)
        probabilities, labels = predict(
            find_checkpoint(args.experiment, args.run_id, fold), args.experiment,
            make_loader(csv_path, args, seed), device, f"{args.experiment} OOF fold {fold}",
        )
        if not np.array_equal(labels, source["label"].to_numpy(dtype=np.int64)):
            raise RuntimeError(f"Fold {fold} label order does not match its CSV")
        frame = source[["patient_id", "label"]].copy()
        frame.insert(0, "fold", fold)
        frame["prediction"] = probabilities.argmax(axis=1)
        frame[PROBABILITY_COLUMNS] = probabilities
        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)
    if len(predictions) != 2386 or predictions["patient_id"].duplicated().any():
        raise RuntimeError("OOF predictions are incomplete or contain duplicate patients")
    labels = predictions["label"].to_numpy(dtype=np.int64)
    probabilities = predictions[PROBABILITY_COLUMNS].to_numpy()
    metrics = summarize(labels, probabilities, args, f"{args.experiment} OOF")
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    return metrics


def evaluate_test(args, device):
    source = pd.read_csv(args.csv)
    loader = make_loader(args.csv, args, args.bootstrap_seed)
    fold_probabilities = []
    labels = None
    for fold, seed in enumerate(SEEDS):
        probabilities, current_labels = predict(
            find_checkpoint(args.experiment, args.run_id, fold), args.experiment,
            loader, device, f"{args.experiment} test fold {fold}",
        )
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise RuntimeError("Test label order changed between folds")
        fold_probabilities.append(probabilities)
    if not np.array_equal(labels, source["label"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Test label order does not match its CSV")

    probabilities = np.mean(np.stack(fold_probabilities), axis=0)
    metrics = summarize(labels, probabilities, args, f"{args.experiment} test ensemble")
    predictions = source[["patient_id", "label"]].copy()
    predictions["prediction"] = probabilities.argmax(axis=1)
    predictions[PROBABILITY_COLUMNS] = probabilities
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "classification_report.txt").write_text(
        classification_report(
            labels, probabilities.argmax(axis=1),
            target_names=["birads0", "birads123", "birads45"],
            digits=4, zero_division=0,
        ),
        encoding="utf-8",
    )
    return metrics


def main():
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly select --device cpu if intended")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    metrics = evaluate_oof(args, device) if args.mode == "oof" else evaluate_test(args, device)
    result = {
        "experiment": args.experiment,
        "mode": args.mode,
        "run_id": args.run_id,
        "configuration": EXPERIMENTS[args.experiment],
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
