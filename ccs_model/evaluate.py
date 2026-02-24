"""Evaluation metrics for CCS model."""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from ccs_model.config import BLOOM_LEVELS, FIGURES_DIR, NUM_LEVELS


def compute_metrics(true_levels, pred_levels, pred_scores=None):
    """Compute all CCS evaluation metrics.

    Args:
        true_levels: list of int (1-6)
        pred_levels: list of int (1-6), rounded predictions
        pred_scores: list of float (1.0-6.0), raw regression scores (optional)

    Returns:
        dict of metrics
    """
    true_arr = np.array(true_levels)
    pred_arr = np.array(pred_levels)

    metrics = {}

    # Exact accuracy
    metrics["exact_accuracy"] = accuracy_score(true_arr, pred_arr)

    # Adjacent accuracy (within ±1)
    metrics["adjacent_accuracy"] = np.mean(np.abs(true_arr - pred_arr) <= 1)

    # MAE
    metrics["mae"] = mean_absolute_error(true_arr, pred_arr)

    # Spearman rho (ordinal correlation)
    if len(set(true_arr)) > 1 and len(set(pred_arr)) > 1:
        rho, p_rho = spearmanr(true_arr, pred_arr)
        metrics["spearman_rho"] = rho
        metrics["spearman_p"] = p_rho
    else:
        metrics["spearman_rho"] = 0.0
        metrics["spearman_p"] = 1.0

    # Pearson r (on raw scores if available)
    if pred_scores is not None:
        scores_arr = np.array(pred_scores)
        if len(set(true_arr)) > 1 and len(set(scores_arr)) > 1:
            r, p_r = pearsonr(true_arr.astype(float), scores_arr)
            metrics["pearson_r"] = r
            metrics["pearson_p"] = p_r
        else:
            metrics["pearson_r"] = 0.0
            metrics["pearson_p"] = 1.0

    # Per-level F1
    f1_per = f1_score(true_arr, pred_arr, labels=list(range(1, NUM_LEVELS + 1)),
                      average=None, zero_division=0)
    for i, f1 in enumerate(f1_per, 1):
        metrics[f"f1_L{i}"] = f1

    # Macro/weighted F1
    metrics["f1_macro"] = f1_score(true_arr, pred_arr, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(true_arr, pred_arr, average="weighted", zero_division=0)

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(
        true_arr, pred_arr, labels=list(range(1, NUM_LEVELS + 1))
    ).tolist()

    return metrics


def print_metrics(metrics, title="Evaluation Results"):
    """Pretty-print metrics to console."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Exact Accuracy:    {metrics['exact_accuracy']:.4f}")
    print(f"  Adjacent Accuracy: {metrics['adjacent_accuracy']:.4f}")
    print(f"  MAE:               {metrics['mae']:.4f}")
    print(f"  Spearman ρ:        {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.4e})")
    if "pearson_r" in metrics:
        print(f"  Pearson r:         {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
    print(f"  F1 Macro:          {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted:       {metrics['f1_weighted']:.4f}")
    print(f"\n  Per-Level F1:")
    for i in range(1, NUM_LEVELS + 1):
        name = BLOOM_LEVELS[i]
        print(f"    L{i} {name:12s}: {metrics[f'f1_L{i}']:.4f}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(metrics, save_path=None, title="CCS Confusion Matrix"):
    """Plot 6×6 confusion matrix heatmap."""
    cm = np.array(metrics["confusion_matrix"])
    labels = [f"L{i}\n{BLOOM_LEVELS[i][:4]}" for i in range(1, NUM_LEVELS + 1)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Level", fontsize=12)
    ax.set_ylabel("True Level", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    plt.close(fig)


def plot_per_level_f1(metrics, save_path=None, title="CCS Per-Level F1 Score"):
    """Plot bar chart of F1 per Bloom's level."""
    levels = list(range(1, NUM_LEVELS + 1))
    f1_scores = [metrics[f"f1_L{i}"] for i in levels]
    names = [BLOOM_LEVELS[i] for i in levels]
    colors = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, f1_scores, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=metrics["f1_macro"], color="red", linestyle="--", alpha=0.7, label=f"Macro F1: {metrics['f1_macro']:.3f}")
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved per-level F1 to {save_path}")
    plt.close(fig)


def plot_training_curves(train_losses, val_losses, val_accuracies, save_path=None):
    """Plot training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, val_losses, "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accuracies, "g-o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Exact Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    plt.close(fig)


def save_results(results, save_path):
    """Save results dict to JSON (handle numpy types)."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Saved results to {save_path}")
