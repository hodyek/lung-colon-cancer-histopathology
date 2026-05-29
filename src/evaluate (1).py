"""
evaluate.py
-----------
Model evaluation: confusion matrix, ROC curves, and per-class metrics.

Functions:
    get_predictions       — Run inference and collect predictions and labels.
    compute_metrics       — Compute accuracy, AUC-ROC, F1, sensitivity, specificity.
    plot_confusion_matrix — Plot and save a normalised confusion matrix.
    plot_roc_curves       — Plot and save per-class ROC curves.
    overfitting_report    — Print a train vs test accuracy gap analysis.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Prediction collection ─────────────────────────────────────────────────────

def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference over an entire DataLoader.

    Returns
    -------
    y_true  : shape (N,)  — integer ground-truth labels
    y_pred  : shape (N,)  — integer predicted labels
    y_prob  : shape (N, C) — softmax probabilities for each class
    """
    model.eval()
    all_labels: List[int] = []
    all_preds:  List[int] = []
    all_probs:  List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Compute overall and per-class evaluation metrics.

    Returns
    -------
    dict with keys:
        accuracy    — float
        macro_f1    — float
        macro_auc   — float
        per_class   — dict of {class_name: {sensitivity, specificity, f1, auc}}
    """
    num_classes = len(class_names)
    accuracy  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro")
    macro_auc = roc_auc_score(
        label_binarize(y_true, classes=list(range(num_classes))),
        y_prob, average="macro", multi_class="ovr"
    )

    per_class: Dict = {}
    cm = confusion_matrix(y_true, y_pred)

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        y_bin = (y_true == i).astype(int)
        class_auc = roc_auc_score(y_bin, y_prob[:, i])
        class_f1  = f1_score(y_true, y_pred, labels=[i], average="macro")

        per_class[name] = {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "f1":          round(class_f1, 4),
            "auc":         round(class_auc, 4),
        }

    return {
        "accuracy":  round(accuracy, 4),
        "macro_f1":  round(macro_f1, 4),
        "macro_auc": round(macro_auc, 4),
        "per_class": per_class,
    }


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str,
) -> None:
    """
    Plot and save a normalised confusion matrix.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground-truth and predicted integer labels.
    class_names : list of str
        Human-readable class names in label order.
    model_name : str
        Used in the figure title.
    save_path : str
        Full path where the PNG is saved.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax,
    )
    ax.set_title(f"{model_name} — Normalised Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {save_path}")


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str,
) -> None:
    """
    Plot and save per-class ROC curves with AUC scores in the legend.

    Parameters
    ----------
    y_true  : shape (N,)   — integer ground-truth labels
    y_prob  : shape (N, C) — softmax probabilities
    class_names : list of str
    model_name : str
    save_path : str
    """
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10.colors

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"{model_name} — Per-Class ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {save_path}")


# ── Overfitting report ────────────────────────────────────────────────────────

def overfitting_report(
    train_acc: float,
    test_acc: float,
    model_name: str,
    threshold: float = 0.05,
) -> None:
    """
    Print a brief overfitting analysis comparing train and test accuracy.

    A gap larger than threshold (default 5 pp) is flagged as a warning.
    A negative gap (test > train) is expected with dropout and augmentation
    and is reported as OK.

    Parameters
    ----------
    train_acc, test_acc : float
        Final training accuracy and test accuracy (0–1 scale).
    model_name : str
        Printed in the report header.
    threshold : float
        Gap size above which a WARNING is printed.
    """
    gap = train_acc - test_acc
    status = (
        "WARNING — gap > {:.0%}".format(threshold) if gap > threshold
        else "NOTE — test > train (healthy with dropout/augmentation)" if gap < 0
        else "OK — gap within acceptable range"
    )

    print("=" * 55)
    print(f"  {model_name} — OVERFITTING ANALYSIS")
    print("=" * 55)
    print(f"  Final Train Accuracy : {train_acc:.4f}")
    print(f"  Test Accuracy        : {test_acc:.4f}")
    print(f"  Gap                  : {gap:.4f}")
    print(f"  Status               : {status}")
    print("=" * 55)
