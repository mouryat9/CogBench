"""Random baseline — uniform random assignment of Bloom's levels."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ccs_model.config import NUM_LEVELS, SEED


def predict_random(questions, seed=SEED):
    """Predict random Bloom's levels (1-6) with uniform distribution."""
    rng = np.random.RandomState(seed)
    return rng.randint(1, NUM_LEVELS + 1, size=len(questions)).tolist()


def evaluate_random(questions, n_trials=100, seed=SEED):
    """Average random baseline over multiple trials for stable estimate."""
    all_true = [q["bloom_level"] for q in questions]
    all_preds = []

    for trial in range(n_trials):
        rng = np.random.RandomState(seed + trial)
        preds = rng.randint(1, NUM_LEVELS + 1, size=len(questions)).tolist()
        all_preds.append(preds)

    # Average metrics across trials
    from ccs_model.evaluate import compute_metrics
    metrics_list = [compute_metrics(all_true, p) for p in all_preds]

    avg_metrics = {}
    for key in ["exact_accuracy", "adjacent_accuracy", "mae", "f1_macro"]:
        values = [m[key] for m in metrics_list]
        avg_metrics[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    return avg_metrics


if __name__ == "__main__":
    from ccs_model.data_loader import load_labeled_questions
    questions = load_labeled_questions()
    results = evaluate_random(questions)
    print("Random Baseline (100 trials):")
    for k, v in results.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
