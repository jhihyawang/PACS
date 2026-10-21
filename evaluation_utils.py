"""Shared metrics and plotting utilities for retained experiment evaluators."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


CLASS_NAMES = ["birads0", "birads123", "birads45"]


def clinical_metrics(labels, predictions):
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    actionable_total = cm[0].sum() + cm[2].sum()
    actionable_correct = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]
    return {
        "actionable_sensitivity": actionable_correct / max(actionable_total, 1),
        "actionable_specificity": cm[1, 1] / max(cm[1].sum(), 1),
        "birads0_recall": cm[0, 0] / max(cm[0].sum(), 1),
        "birads45_recall": cm[2, 2] / max(cm[2].sum(), 1),
        "birads0_to_123": int(cm[0, 1]),
        "birads45_to_123": int(cm[2, 1]),
    }


def calculate_metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
    one_hot = np.eye(3, dtype=np.int64)[labels]
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "auroc": float(roc_auc_score(one_hot[:, index], probabilities[:, index])),
            "auprc": float(average_precision_score(one_hot[:, index], probabilities[:, index])),
        }
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
    }
    metrics.update(clinical_metrics(labels, predictions))
    return metrics


def bootstrap_intervals(labels, probabilities, samples, seed):
    rng = np.random.default_rng(seed)
    keys = [
        "accuracy", "macro_f1", "weighted_f1", "birads0_recall",
        "birads45_recall", "actionable_sensitivity", "actionable_specificity",
    ]
    values = {key: [] for key in keys}
    for _ in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        sample_labels = labels[indices]
        sample_predictions = probabilities[indices].argmax(axis=1)
        metrics = {
            "accuracy": accuracy_score(sample_labels, sample_predictions),
            "macro_f1": f1_score(sample_labels, sample_predictions, labels=[0, 1, 2], average="macro", zero_division=0),
            "weighted_f1": f1_score(sample_labels, sample_predictions, labels=[0, 1, 2], average="weighted", zero_division=0),
        }
        metrics.update(clinical_metrics(sample_labels, sample_predictions))
        for key in keys:
            values[key].append(metrics[key])
    return {
        key: {
            "lower": float(np.percentile(metric_values, 2.5)),
            "upper": float(np.percentile(metric_values, 97.5)),
        }
        for key, metric_values in values.items()
    }


def save_confusion_matrix(labels, probabilities, path, title):
    cm = confusion_matrix(labels, probabilities.argmax(axis=1), labels=[0, 1, 2])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
