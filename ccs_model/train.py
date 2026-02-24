"""Phase 1: 5-fold cross-validation with BERT-base end-to-end fine-tuning.

Architecture: BERT-base-uncased (all layers trainable) + MLP head
Differential learning rates: encoder gets 2e-5, head gets 5x = 1e-4
Training: ~2-3 min per fold on RTX 2080 Ti (11GB)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccs_model.config import (
    BATCH_SIZE, LEARNING_RATE, HEAD_LR_MULTIPLIER, WEIGHT_DECAY,
    NUM_EPOCHS, EARLY_STOPPING_PATIENCE, GPU_DEVICE,
    SEED, OUTPUT_DIR, FIGURES_DIR, N_FOLDS, MAX_SEQ_LEN
)
from ccs_model.data_loader import load_labeled_questions, get_tokenizer, get_kfold_splits, BloomDataset
from ccs_model.model import CCSModel, count_parameters
from ccs_model.evaluate import (
    compute_metrics, print_metrics, plot_confusion_matrix,
    plot_per_level_f1, plot_training_curves, save_results
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, mode, grad_accum=2):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["label"].to(device)

        output = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(output, labels) / grad_accum

        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        n_batches += 1

    scheduler.step()
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_true = []
    all_scores = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["label"].to(device)

        output = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(output, labels)
        total_loss += loss.item()
        n_batches += 1

        if mode == "regression":
            scores = output.cpu().numpy()
            preds = np.clip(np.round(scores), 1, 6).astype(int).tolist()
            true = labels.cpu().numpy().astype(int).tolist()
            all_scores.extend(scores.tolist())
        else:
            preds = (output.argmax(dim=1).cpu().numpy() + 1).tolist()
            true = (labels.cpu().numpy() + 1).tolist()
            all_scores = None

        all_preds.extend(preds)
        all_true.extend(true)

    avg_loss = total_loss / n_batches
    metrics = compute_metrics(all_true, all_preds, all_scores if mode == "regression" else None)
    return avg_loss, metrics


def train_fold(fold_idx, train_qs, val_qs, tokenizer, device, mode="regression"):
    """Train one fold end-to-end."""
    print(f"\n  FOLD {fold_idx + 1}/{N_FOLDS} — {mode}")
    print(f"  Train: {len(train_qs)}, Val: {len(val_qs)}")

    # Datasets
    train_ds = BloomDataset(train_qs, tokenizer, max_len=MAX_SEQ_LEN, mode=mode)
    val_ds = BloomDataset(val_qs, tokenizer, max_len=MAX_SEQ_LEN, mode=mode)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    # Model
    model = CCSModel(mode=mode).to(device)
    trainable, total = count_parameters(model)
    print(f"  Params: {trainable:,} trainable / {total:,} total")

    # Optimizer with differential LR
    head_lr = LEARNING_RATE * HEAD_LR_MULTIPLIER
    param_groups = model.get_optimizer_groups(
        encoder_lr=LEARNING_RATE, head_lr=head_lr, weight_decay=WEIGHT_DECAY
    )
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LEARNING_RATE * 0.01
    )

    if mode == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    best_mae = float("inf")
    best_state = None
    best_metrics = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 criterion, device, mode, grad_accum=2)
        train_losses.append(train_loss)

        # Validate
        val_loss, metrics = evaluate(model, val_loader, criterion, device, mode)
        val_losses.append(val_loss)
        val_accuracies.append(metrics["exact_accuracy"])

        epoch_time = time.time() - t0

        # Early stopping on MAE
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_metrics = metrics
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"    Epoch {epoch+1:3d}/{NUM_EPOCHS} ({epoch_time:.1f}s) | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Acc: {metrics['exact_accuracy']:.4f} | Adj: {metrics['adjacent_accuracy']:.4f} | "
                  f"MAE: {metrics['mae']:.4f}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Save best checkpoint
    fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    path = os.path.join(fold_dir, f"best_model_{mode}.pt")
    torch.save(best_state, path)
    print(f"  Saved: {path}")
    print(f"  Best MAE: {best_mae:.4f} | Acc: {best_metrics['exact_accuracy']:.4f}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return best_metrics, best_state, train_losses, val_losses, val_accuracies


def run_phase1(mode="regression"):
    """Run full 5-fold cross-validation."""
    set_seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (GPU {GPU_DEVICE})")

    # Load data
    questions = load_labeled_questions()
    tokenizer = get_tokenizer()

    # Get fold splits
    splits = get_kfold_splits(questions)

    all_fold_metrics = []
    all_true = []
    all_pred = []

    t_start = time.time()

    for split in splits:
        fold_metrics, best_state, t_loss, v_loss, v_acc = train_fold(
            split["fold"], split["train"], split["val"],
            tokenizer, device, mode
        )
        all_fold_metrics.append(fold_metrics)

        # Collect predictions from best model for aggregate metrics
        model = CCSModel(mode=mode).to(device)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

        val_ds = BloomDataset(split["val"], tokenizer, max_len=MAX_SEQ_LEN, mode=mode)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                output = model(input_ids, attention_mask, token_type_ids)

                if mode == "regression":
                    scores = output.cpu().numpy()
                    preds = np.clip(np.round(scores), 1, 6).astype(int).tolist()
                    true = batch["label"].numpy().astype(int).tolist()
                else:
                    preds = (output.argmax(dim=1).cpu().numpy() + 1).tolist()
                    true = (batch["label"].numpy() + 1).tolist()

                all_true.extend(true)
                all_pred.extend(preds)

        del model
        torch.cuda.empty_cache()

        plot_training_curves(t_loss, v_loss, v_acc,
                             os.path.join(FIGURES_DIR, f"ccs_training_fold{split['fold']}_{mode}.png"))

    total_time = time.time() - t_start

    # Aggregate
    aggregate = compute_metrics(all_true, all_pred)
    metric_keys = ["exact_accuracy", "adjacent_accuracy", "mae", "spearman_rho", "f1_macro", "f1_weighted"]
    per_fold = {}
    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics]
        per_fold[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                         "per_fold": [float(v) for v in values]}

    print(f"\n{'#'*60}")
    print(f"  PHASE 1 RESULTS — {mode.upper()} — {N_FOLDS}-FOLD CV")
    print(f"  Training: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'#'*60}")
    print_metrics(aggregate, f"Aggregate (all {len(all_true)} predictions)")

    print("Per-fold summary:")
    for key in metric_keys:
        print(f"  {key:20s}: {per_fold[key]['mean']:.4f} ± {per_fold[key]['std']:.4f}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_confusion_matrix(aggregate, os.path.join(FIGURES_DIR, f"ccs_confusion_matrix_{mode}.png"),
                          f"CCS {mode.title()} — {N_FOLDS}-Fold CV Confusion Matrix")
    plot_per_level_f1(aggregate, os.path.join(FIGURES_DIR, f"ccs_per_level_f1_{mode}.png"),
                      f"CCS {mode.title()} — Per-Level F1 ({N_FOLDS}-Fold CV)")

    results = {
        "mode": mode,
        "model": "bert-base-uncased",
        "n_folds": N_FOLDS,
        "n_samples": len(questions),
        "training_time_seconds": total_time,
        "aggregate_metrics": aggregate,
        "per_fold_summary": per_fold,
        "per_fold_metrics": all_fold_metrics,
    }
    save_results(results, os.path.join(OUTPUT_DIR, f"phase1_results_{mode}.json"))
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CCS Phase 1: 5-fold CV Training")
    parser.add_argument("--mode", choices=["regression", "classification", "both"], default="both")
    args = parser.parse_args()

    modes = ["regression", "classification"] if args.mode == "both" else [args.mode]

    all_results = {}
    for mode in modes:
        print(f"\n{'*'*60}")
        print(f"  Starting {mode.upper()} training")
        print(f"{'*'*60}")
        all_results[mode] = run_phase1(mode)

    if len(modes) == 2:
        print(f"\n{'#'*60}")
        print(f"  COMPARISON: Regression vs Classification")
        print(f"{'#'*60}")
        for key in ["exact_accuracy", "adjacent_accuracy", "mae", "spearman_rho", "f1_macro"]:
            reg = all_results["regression"]["aggregate_metrics"][key]
            cls = all_results["classification"]["aggregate_metrics"][key]
            winner = "REG" if (reg > cls if key != "mae" else reg < cls) else "CLS"
            print(f"  {key:20s}: Reg={reg:.4f} | Cls={cls:.4f} | Winner: {winner}")
