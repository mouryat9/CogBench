"""CogBench benchmark metrics.

Computes per-model and per-level metrics from scored generation files.

Key metrics:
  - Level Accuracy: % of generated questions at the correct Bloom's level
  - Adjacent Accuracy: % within ±1 level
  - Mean Absolute Error: average |predicted - target| level distance
  - Cognitive Control Score (CCS-Control): composite metric
  - Level Distribution Fidelity: KL divergence between target and actual
"""

import os
import sys
import json
import glob
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.config import BLOOM_LEVELS, RESULTS_DIR


def compute_benchmark_metrics(generations: list) -> dict:
    """Compute benchmark metrics from a list of scored generation records.

    Each record should have: level (target), ccs_level (predicted), is_correct, is_adjacent.
    """
    # Filter to valid (scored) records
    valid = [g for g in generations if g.get("ccs_level") is not None]
    if not valid:
        return {"n_valid": 0, "error": "No valid scored generations"}

    targets = np.array([g["level"] for g in valid])
    predicted = np.array([g["ccs_level"] for g in valid])
    errors = predicted - targets

    n = len(valid)
    n_correct = int(np.sum(predicted == targets))
    n_adjacent = int(np.sum(np.abs(errors) <= 1))

    # Core metrics
    level_accuracy = n_correct / n
    adjacent_accuracy = n_adjacent / n
    mae = float(np.mean(np.abs(errors)))

    # Cognitive Control Score: weighted combination
    # CCS-Control = 0.5 * LevelAcc + 0.3 * AdjAcc + 0.2 * (1 - MAE/5)
    mae_normalized = 1.0 - mae / 5.0  # MAE on [1,6] scale → 0-1 where 1 is perfect
    ccs_control = 0.5 * level_accuracy + 0.3 * adjacent_accuracy + 0.2 * mae_normalized

    # Per-level metrics
    per_level = {}
    for lv in sorted(BLOOM_LEVELS.keys()):
        mask = targets == lv
        if mask.sum() == 0:
            per_level[lv] = {"n": 0, "accuracy": None, "adjacent": None, "mae": None}
            continue
        lv_pred = predicted[mask]
        lv_acc = float(np.mean(lv_pred == lv))
        lv_adj = float(np.mean(np.abs(lv_pred - lv) <= 1))
        lv_mae = float(np.mean(np.abs(lv_pred - lv)))
        per_level[lv] = {
            "n": int(mask.sum()),
            "accuracy": lv_acc,
            "adjacent": lv_adj,
            "mae": lv_mae,
            "mean_predicted": float(np.mean(lv_pred)),
        }

    # Distribution fidelity (KL divergence)
    target_dist = np.array([np.sum(targets == lv) for lv in range(1, 7)], dtype=float)
    pred_dist = np.array([np.sum(predicted == lv) for lv in range(1, 7)], dtype=float)
    # Normalize + smoothing
    target_dist = (target_dist + 1e-8) / target_dist.sum()
    pred_dist = (pred_dist + 1e-8) / pred_dist.sum()
    kl_div = float(np.sum(target_dist * np.log(target_dist / pred_dist)))

    # Mean confidence
    confidences = [g["ccs_confidence"] for g in valid if g.get("ccs_confidence") is not None]
    mean_confidence = float(np.mean(confidences)) if confidences else None

    return {
        "n_valid": n,
        "n_correct": n_correct,
        "n_adjacent": n_adjacent,
        "level_accuracy": level_accuracy,
        "adjacent_accuracy": adjacent_accuracy,
        "mae": mae,
        "ccs_control": ccs_control,
        "kl_divergence": kl_div,
        "mean_confidence": mean_confidence,
        "per_level": per_level,
        "target_distribution": target_dist.tolist(),
        "predicted_distribution": pred_dist.tolist(),
    }


def compute_per_subject_metrics(generations: list) -> dict:
    """Compute metrics broken down by subject."""
    by_subject = defaultdict(list)
    for g in generations:
        if g.get("ccs_level") is not None:
            by_subject[g["subject"]].append(g)

    results = {}
    for subject, gens in sorted(by_subject.items()):
        results[subject] = compute_benchmark_metrics(gens)
    return results


def compute_per_template_metrics(generations: list) -> dict:
    """Compute metrics broken down by prompt template."""
    by_template = defaultdict(list)
    for g in generations:
        if g.get("ccs_level") is not None:
            by_template[g["template"]].append(g)

    results = {}
    for template, gens in sorted(by_template.items()):
        results[template] = compute_benchmark_metrics(gens)
    return results


def print_model_summary(model: str, metrics: dict):
    """Print a formatted summary for one model."""
    print(f"\n  {model}")
    print(f"  {'─' * 55}")
    print(f"  Level Accuracy:     {metrics['level_accuracy']:.1%}")
    print(f"  Adjacent Accuracy:  {metrics['adjacent_accuracy']:.1%}")
    print(f"  MAE:                {metrics['mae']:.3f}")
    print(f"  CCS-Control:        {metrics['ccs_control']:.3f}")
    print(f"  KL Divergence:      {metrics['kl_divergence']:.3f}")
    print(f"  N valid:            {metrics['n_valid']}")

    print(f"\n  Per-Level Accuracy:")
    print(f"  {'Level':<12s} {'N':>5s} {'Acc':>7s} {'Adj':>7s} {'MAE':>7s} {'AvgPred':>8s}")
    print(f"  {'─' * 49}")
    for lv in sorted(BLOOM_LEVELS.keys()):
        pl = metrics["per_level"].get(lv, {})
        if pl.get("n", 0) == 0:
            continue
        print(f"  L{lv} {BLOOM_LEVELS[lv]:<9s} {pl['n']:>5d} "
              f"{pl['accuracy']:>6.1%} {pl['adjacent']:>6.1%} "
              f"{pl['mae']:>6.3f} {pl['mean_predicted']:>7.2f}")


def load_and_analyze(results_dir=RESULTS_DIR):
    """Load all scored generation files and produce summary."""
    pattern = os.path.join(results_dir, "gen_*_scored.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No scored files found in {results_dir}")
        return

    all_results = {}

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        model = data.get("model", "unknown")
        template = data.get("template", "unknown")
        key = f"{model} ({template})"
        generations = data.get("generations", [])
        metrics = compute_benchmark_metrics(generations)
        all_results[key] = metrics
        print_model_summary(key, metrics)

    # Summary table
    print(f"\n{'=' * 75}")
    print(f"  COGBENCH BENCHMARK SUMMARY")
    print(f"{'=' * 75}")
    print(f"\n  {'Model':<35s} {'LvlAcc':>7s} {'AdjAcc':>7s} {'MAE':>6s} {'CCS-C':>6s} {'KL':>6s}")
    print(f"  {'─' * 68}")
    for key, m in sorted(all_results.items(), key=lambda x: -x[1].get("level_accuracy", 0)):
        print(f"  {key:<35s} {m['level_accuracy']:>6.1%} {m['adjacent_accuracy']:>6.1%} "
              f"{m['mae']:>5.3f} {m['ccs_control']:>5.3f} {m['kl_divergence']:>5.3f}")

    return all_results


if __name__ == "__main__":
    load_and_analyze()
