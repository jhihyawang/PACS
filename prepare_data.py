#!/usr/bin/env python3
"""Build patient-level train/validation/test CSVs and five development folds."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")
CATEGORY_TO_LABEL = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def discover_examinations(data_root):
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    records = []
    skipped = []
    category_pattern = re.compile(r"^Category\s+([0-9]+)$", re.IGNORECASE)
    for category_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        match = category_pattern.match(category_dir.name)
        if not match:
            skipped.append(str(category_dir))
            continue
        category = int(match.group(1))
        if category not in CATEGORY_TO_LABEL:
            skipped.append(str(category_dir))
            continue

        for patient_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            paths = {view: patient_dir / f"{view}.jpg" for view in VIEWS}
            missing = [view for view, path in paths.items() if not path.is_file()]
            if missing:
                raise RuntimeError(
                    f"Incomplete examination {patient_dir}: missing {', '.join(missing)}"
                )
            records.append(
                {
                    **{
                        view: paths[view].relative_to(data_root).as_posix()
                        for view in VIEWS
                    },
                    "label": CATEGORY_TO_LABEL[category],
                    "patient_id": patient_dir.relative_to(data_root).as_posix(),
                }
            )

    frame = pd.DataFrame(records, columns=[*VIEWS, "label", "patient_id"])
    if frame.empty:
        raise RuntimeError(f"No complete examinations found under {data_root}")
    if frame["patient_id"].duplicated().any():
        duplicates = frame.loc[frame["patient_id"].duplicated(), "patient_id"].tolist()
        raise RuntimeError(f"Duplicate patient IDs found: {duplicates[:5]}")
    return frame, skipped


def stratified_splits(frame, seed):
    total = len(frame)
    test_size = round(total * 0.20)
    validation_size = round(total * 0.10)
    development, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=frame["label"],
    )
    train, validation = train_test_split(
        development,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=development["label"],
    )
    return tuple(split.reset_index(drop=True) for split in (train, validation, test))


def class_counts(frame):
    counts = frame["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    return {"birads0": int(counts[0]), "birads1_3": int(counts[1]), "birads4_5": int(counts[2])}


def write_primary_splits(train, validation, test, output_dir):
    split_dir = output_dir / "three_classes"
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train_labels.csv", index=False)
    validation.to_csv(split_dir / "val_labels.csv", index=False)
    test.to_csv(split_dir / "test_labels.csv", index=False)


def write_folds(train, validation, output_dir, n_splits, seed):
    development = pd.concat([train, validation], ignore_index=True)
    fold_root = output_dir / "e17_folds"
    fold_root.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    summaries = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(development, development["label"])
    ):
        fold_train = development.iloc[train_indices].reset_index(drop=True)
        fold_validation = development.iloc[validation_indices].reset_index(drop=True)
        fold_dir = fold_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_train.to_csv(fold_dir / "train_labels.csv", index=False)
        fold_validation.to_csv(fold_dir / "val_labels.csv", index=False)
        summaries.append(
            {
                "fold": fold,
                "train": {"total": len(fold_train), **class_counts(fold_train)},
                "validation": {"total": len(fold_validation), **class_counts(fold_validation)},
            }
        )
    return summaries


def main():
    args = parse_args()
    frame, skipped = discover_examinations(args.data_root)
    train, validation, test = stratified_splits(frame, args.seed)
    write_primary_splits(train, validation, test, args.output_dir)
    fold_summaries = write_folds(
        train, validation, args.output_dir, args.folds, args.seed
    )

    manifest = {
        "data_root": str(args.data_root),
        "seed": args.seed,
        "category_mapping": {
            "BI-RADS 0": [0], "BI-RADS 1-3": [1, 2, 3], "BI-RADS 4-5": [4, 5]
        },
        "all": {"total": len(frame), **class_counts(frame)},
        "train": {"total": len(train), **class_counts(train)},
        "validation": {"total": len(validation), **class_counts(validation)},
        "test": {"total": len(test), **class_counts(test)},
        "folds": fold_summaries,
        "skipped_directories": skipped,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(pd.DataFrame([
        {"split": "train", "total": len(train), **class_counts(train)},
        {"split": "validation", "total": len(validation), **class_counts(validation)},
        {"split": "test", "total": len(test), **class_counts(test)},
    ]).to_string(index=False))
    print(f"Saved CSV files to {args.output_dir}")
    print(f"Saved split manifest to {manifest_path}")


if __name__ == "__main__":
    main()
