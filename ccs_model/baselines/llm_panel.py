"""LLM panel baseline — use local Ollama models for zero-shot Bloom's classification.

Runs each model sequentially (one at a time) to avoid GPU memory conflicts.
Reports per-model metrics AND panel majority vote.
Saves predictions to disk for resumability.
"""

import sys, os
import json
import re
import time
import requests
import numpy as np
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ccs_model.config import BLOOM_LEVELS, N_FOLDS, SEED, OUTPUT_DIR
from ccs_model.data_loader import load_labeled_questions, get_kfold_splits
from ccs_model.evaluate import compute_metrics, print_metrics

OLLAMA_URL = "http://localhost:11434/api/generate"

# 4-judge panel (same as thesis ablation experiment)
LLM_PANEL_MODELS = [
    "qwen2.5:14b",
    "llama3.1:8b",
    "gemma3:12b",
    "phi4:14b",
]

# Additional models available for extended comparison
EXTENDED_MODELS = [
    "deepseek-r1:14b",
    "mistral-nemo:12b",
]

CLASSIFICATION_PROMPT = """Classify the following question according to Bloom's Taxonomy Revised (Anderson & Krathwohl, 2001).

The 6 levels are:
1. Remember — Retrieve knowledge from memory (define, list, recall, identify, name, state)
2. Understand — Construct meaning from information (explain, describe, summarize, interpret, paraphrase)
3. Apply — Use a procedure in a given situation (calculate, solve, demonstrate, implement, use)
4. Analyze — Break into parts, find relationships (compare, contrast, examine, differentiate, categorize)
5. Evaluate — Make judgments based on criteria (justify, critique, assess, argue, defend, judge)
6. Create — Put elements together into a new whole (design, propose, construct, develop, formulate)

Question: {question}

Respond with ONLY the level number (1-6). Nothing else."""


def unload_model(model):
    """Explicitly unload a model from Ollama GPU memory."""
    try:
        requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": "",
            "keep_alive": 0,
        }, timeout=10)
    except Exception:
        pass


def classify_single(question_text, model, timeout=60):
    """Classify a single question using Ollama. Returns (level, raw_response)."""
    prompt = CLASSIFICATION_PROMPT.format(question=question_text)

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 10},
        }, timeout=timeout)
        resp.raise_for_status()
        text = resp.json()["response"].strip()

        # Extract first digit 1-6
        match = re.search(r'[1-6]', text)
        if match:
            return int(match.group()), text
        return None, text  # No valid level found

    except Exception as e:
        return None, f"ERROR: {e}"


def classify_all_with_model(questions, model, save_path=None):
    """Classify all questions with a single model. Saves progress incrementally.

    Returns list of predictions (int 1-6, or None for failures).
    """
    # Load existing predictions if resuming
    predictions = {}
    if save_path and os.path.exists(save_path):
        with open(save_path) as f:
            saved = json.load(f)
            predictions = {str(k): v for k, v in saved.get("predictions", {}).items()}
        print(f"  Resuming from {len(predictions)} existing predictions")

    n_total = len(questions)
    n_done = len(predictions)
    n_failed = 0
    t_start = time.time()

    # Warm up the model with first request (triggers loading)
    if n_done == 0:
        print(f"  Warming up {model}...")
        classify_single("What is 2+2?", model, timeout=120)

    for i, q in enumerate(questions):
        qid = str(i)
        if qid in predictions:
            continue

        level, raw = classify_single(q["question_text"], model)
        if level is None:
            n_failed += 1
            level = 2  # Default to Understand (most common level)

        predictions[qid] = {
            "predicted_level": level,
            "raw_response": raw,
            "question_id": q.get("question_id", f"q_{i}"),
            "true_level": q["bloom_level"],
        }

        n_done += 1

        # Progress report every 50 questions
        if n_done % 50 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"  [{model}] {n_done}/{n_total} "
                  f"({n_done/n_total*100:.0f}%) | "
                  f"{rate:.1f} q/s | ETA: {eta:.0f}s | "
                  f"Failed: {n_failed}")

            # Save incrementally
            if save_path:
                _save_predictions(save_path, model, predictions)

    # Final save
    if save_path:
        _save_predictions(save_path, model, predictions)

    elapsed = time.time() - t_start
    print(f"  [{model}] Done: {n_total} questions in {elapsed:.0f}s "
          f"({elapsed/n_total:.1f}s/q, {n_failed} failures)")

    # Unload model to free GPU memory
    print(f"  Unloading {model}...")
    unload_model(model)
    time.sleep(2)  # Let GPU memory settle

    # Return ordered prediction list
    return [predictions[str(i)]["predicted_level"] for i in range(n_total)]


def _save_predictions(path, model, predictions):
    """Save predictions to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "model": model,
            "n_predictions": len(predictions),
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
        }, f, indent=2)


def run_llm_panel(questions=None, models=None, output_dir=None):
    """Run full LLM panel evaluation on all labeled questions.

    Runs each model one at a time, saves per-model predictions,
    then computes per-model metrics and panel majority vote.
    """
    if questions is None:
        questions = load_labeled_questions()
    models = models or LLM_PANEL_MODELS
    output_dir = output_dir or os.path.join(OUTPUT_DIR, "llm_panel_results")
    os.makedirs(output_dir, exist_ok=True)

    true_levels = [q["bloom_level"] for q in questions]
    all_model_preds = {}
    all_model_metrics = {}

    print(f"\n{'='*60}")
    print(f"  LLM Panel Baseline — {len(models)} models × {len(questions)} questions")
    print(f"{'='*60}")

    t_total_start = time.time()

    for model in models:
        print(f"\n--- Model: {model} ---")
        save_path = os.path.join(output_dir, f"preds_{model.replace(':', '_').replace('.', '_')}.json")

        preds = classify_all_with_model(questions, model, save_path)
        all_model_preds[model] = preds

        # Per-model metrics
        metrics = compute_metrics(true_levels, preds)
        all_model_metrics[model] = metrics
        print(f"  Results: Acc={metrics['exact_accuracy']:.4f}, "
              f"Adj={metrics['adjacent_accuracy']:.4f}, "
              f"MAE={metrics['mae']:.4f}, F1={metrics['f1_macro']:.4f}")

    # Panel majority vote
    print(f"\n--- Panel Majority Vote ({len(models)} models) ---")
    panel_preds = []
    for i in range(len(questions)):
        votes = [all_model_preds[m][i] for m in models]
        counter = Counter(votes)
        panel_preds.append(counter.most_common(1)[0][0])

    panel_metrics = compute_metrics(true_levels, panel_preds)

    total_time = time.time() - t_total_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"  LLM PANEL RESULTS SUMMARY")
    print(f"  {len(questions)} questions, {len(models)} models, {total_time:.0f}s total")
    print(f"{'='*60}")

    print(f"\n{'Method':<25s} {'Exact':>7s} {'Adj±1':>7s} {'MAE':>7s} {'F1':>7s} {'Spearman':>9s}")
    print("-" * 65)
    for model in models:
        m = all_model_metrics[model]
        print(f"  {model:<23s} {m['exact_accuracy']:>6.1%} {m['adjacent_accuracy']:>6.1%} "
              f"{m['mae']:>6.3f} {m['f1_macro']:>6.3f} {m.get('spearman_rho', 0):>8.3f}")
    print("-" * 65)
    print(f"  {'Panel (majority vote)':<23s} {panel_metrics['exact_accuracy']:>6.1%} "
          f"{panel_metrics['adjacent_accuracy']:>6.1%} {panel_metrics['mae']:>6.3f} "
          f"{panel_metrics['f1_macro']:>6.3f} {panel_metrics.get('spearman_rho', 0):>8.3f}")

    # Compare with CCS
    print(f"\n  For reference:")
    print(f"  {'CCS (BERT classif.)':<23s} {'82.0%':>7s} {'89.6%':>7s} {'0.356':>7s} {'0.819':>7s} {'0.857':>9s}")
    print(f"  {'TF-IDF + SVM':<23s} {'77.7%':>7s} {'85.3%':>7s} {'0.286':>7s} {'0.776':>7s} {'—':>9s}")

    # Save comprehensive results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(questions),
        "models": models,
        "total_time_seconds": total_time,
        "per_model_metrics": {
            model: {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()
                    if not isinstance(v, np.ndarray)}
            for model, metrics in all_model_metrics.items()
        },
        "panel_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in panel_metrics.items()
                         if not isinstance(v, np.ndarray)},
        "per_model_predictions": {
            model: preds for model, preds in all_model_preds.items()
        },
        "panel_predictions": panel_preds,
        "true_levels": true_levels,
    }

    results_path = os.path.join(output_dir, "llm_panel_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Per-level breakdown for each model
    print(f"\n  Per-Level Accuracy:")
    print(f"  {'Model':<23s}", end="")
    for level in range(1, 7):
        print(f" {'L'+str(level):>6s}", end="")
    print()
    print("  " + "-" * 59)

    for model in models + ["Panel"]:
        preds = panel_preds if model == "Panel" else all_model_preds[model]
        print(f"  {model:<23s}", end="")
        for level in range(1, 7):
            mask = np.array(true_levels) == level
            if mask.sum() > 0:
                level_acc = np.mean(np.array(preds)[mask] == level)
                print(f" {level_acc:>5.1%}", end="")
            else:
                print(f" {'—':>6s}", end="")
        print()

    return results


# K-fold version for strict comparison with CCS
def run_kfold_llm_panel(questions=None, n_folds=N_FOLDS, models=None):
    """Run K-fold evaluation with LLM panel (uses same splits as CCS).

    Since LLMs are zero-shot, K-fold doesn't change results — but this
    lets us report fold-level variance for fair comparison.
    """
    if questions is None:
        questions = load_labeled_questions()
    models = models or LLM_PANEL_MODELS

    # First, get predictions for ALL questions (zero-shot = no training needed)
    output_dir = os.path.join(OUTPUT_DIR, "llm_panel_results")
    results = run_llm_panel(questions, models, output_dir)

    # Now compute fold-level metrics
    splits = get_kfold_splits(questions, n_folds)
    fold_metrics = []

    print(f"\n  K-Fold Breakdown (Panel Majority Vote):")
    for split in splits:
        val_indices = [i for i, q in enumerate(questions) if q in split["val"]]
        val_true = [questions[i]["bloom_level"] for i in val_indices]
        val_pred = [results["panel_predictions"][i] for i in val_indices]

        fm = compute_metrics(val_true, val_pred)
        fold_metrics.append(fm)
        print(f"    Fold {split['fold']}: Acc={fm['exact_accuracy']:.4f}, "
              f"Adj={fm['adjacent_accuracy']:.4f}, MAE={fm['mae']:.4f}")

    # Fold summary
    for key in ["exact_accuracy", "adjacent_accuracy", "mae", "f1_macro"]:
        values = [m[key] for m in fold_metrics]
        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    return results, fold_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Panel Baseline for Bloom's Classification")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to use (default: 4-judge panel)")
    parser.add_argument("--extended", action="store_true",
                        help="Include extended models (deepseek-r1, mistral-nemo)")
    parser.add_argument("--quick", action="store_true",
                        help="Only run on first 50 questions for testing")
    parser.add_argument("--kfold", action="store_true",
                        help="Also compute K-fold breakdown")
    args = parser.parse_args()

    models = args.models or LLM_PANEL_MODELS
    if args.extended:
        models = models + EXTENDED_MODELS

    questions = load_labeled_questions()
    if args.quick:
        questions = questions[:50]
        print(f"Quick mode: using first {len(questions)} questions only")

    if args.kfold and not args.quick:
        results, fold_metrics = run_kfold_llm_panel(questions, models=models)
    else:
        results = run_llm_panel(questions, models=models)
