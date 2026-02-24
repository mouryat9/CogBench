"""Phase 2: Fine-tune CCS on gold human-annotated data (after annotations arrive).

This script is a SCAFFOLD — it will be run once human annotations are complete
and gold labels are exported via collection/export_gold_labels.py.

Expected timeline: ~2 weeks from now (early March 2026).

What it does:
1. Load 739 original labeled questions + ~320 gold train questions = ~1,059 total
2. Load best Phase 1 checkpoint as initialization
3. Fine-tune with same hyperparameters (or adjusted)
4. Evaluate on 80 held-out gold test questions
5. Compare Phase 1 vs Phase 2 performance
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccs_model.config import (
    OUTPUT_DIR, FIGURES_DIR, DATA_FILE, SEED, BATCH_SIZE, GPU_DEVICE,
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, BLOOM_LEVELS
)
from ccs_model.data_loader import load_labeled_questions, get_tokenizer, BloomDataset
from ccs_model.model import CCSRegression, count_parameters
from ccs_model.evaluate import compute_metrics, print_metrics, plot_confusion_matrix, plot_per_level_f1, save_results


# Paths to gold data (created by collection/export_gold_labels.py)
GOLD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
GOLD_TRAIN_FILE = os.path.join(GOLD_DIR, "gold_train.json")
GOLD_TEST_FILE = os.path.join(GOLD_DIR, "gold_test.json")


def load_gold_data():
    """Load gold-annotated train and test sets."""
    if not os.path.exists(GOLD_TRAIN_FILE):
        raise FileNotFoundError(
            f"Gold train file not found at {GOLD_TRAIN_FILE}.\n"
            "Run collection/export_gold_labels.py first after annotations are complete."
        )

    with open(GOLD_TRAIN_FILE) as f:
        gold_train = json.load(f)
    with open(GOLD_TEST_FILE) as f:
        gold_test = json.load(f)

    print(f"Gold train: {len(gold_train)} questions")
    print(f"Gold test:  {len(gold_test)} questions")
    return gold_train, gold_test


def run_phase2(mode="regression"):
    """Phase 2: Fine-tune on combined original + gold labels, evaluate on gold test."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    original = load_labeled_questions()
    gold_train, gold_test = load_gold_data()

    # Combine original + gold train
    combined_train = original + gold_train
    print(f"Combined training set: {len(combined_train)} questions "
          f"({len(original)} original + {len(gold_train)} gold)")

    tokenizer = get_tokenizer()
    train_ds = BloomDataset(combined_train, tokenizer, mode=mode)
    test_ds = BloomDataset(gold_test, tokenizer, mode=mode)

    # Load best Phase 1 checkpoint
    phase1_checkpoint = os.path.join(OUTPUT_DIR, f"fold_0/best_model_{mode}.pt")
    model = CCSRegression() if mode == "regression" else None
    if model is None:
        raise NotImplementedError("Phase 2 classification not yet implemented")

    if os.path.exists(phase1_checkpoint):
        state = torch.load(phase1_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded Phase 1 checkpoint from {phase1_checkpoint}")
    else:
        print("WARNING: No Phase 1 checkpoint found. Training from scratch.")

    model = model.to(device)

    # TODO: Training loop (reuse from train.py)
    # For now, just evaluate Phase 1 model on gold test set
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    from ccs_model.train import evaluate_model
    metrics, true_levels, pred_levels, pred_scores = evaluate_model(model, test_loader, device, mode)

    print_metrics(metrics, "Phase 1 Model → Gold Test Set")

    # Save results
    phase2_dir = os.path.join(OUTPUT_DIR, "phase2")
    os.makedirs(phase2_dir, exist_ok=True)
    save_results({
        "phase": "phase1_on_gold_test",
        "mode": mode,
        "n_train": len(combined_train),
        "n_test": len(gold_test),
        "metrics": metrics,
    }, os.path.join(phase2_dir, "phase1_on_gold_results.json"))

    plot_confusion_matrix(metrics, os.path.join(FIGURES_DIR, "ccs_gold_test_cm.png"),
                          "Phase 1 CCS → Gold Test Set")

    return metrics


if __name__ == "__main__":
    print("CCS Phase 2: Gold Fine-tuning")
    print("=" * 40)
    try:
        run_phase2()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPhase 2 cannot run yet — waiting for human annotations.")
        print("Expected: early March 2026")
