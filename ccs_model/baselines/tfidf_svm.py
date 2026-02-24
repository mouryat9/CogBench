"""TF-IDF + SVM baseline — traditional ML approach."""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from ccs_model.config import SEED, N_FOLDS
from ccs_model.data_loader import load_labeled_questions, get_kfold_splits, format_input_text
from ccs_model.evaluate import compute_metrics, print_metrics


def build_tfidf_svm(C=1.0, kernel="rbf"):
    """Build TF-IDF + SVM pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("svm", SVC(
            C=C,
            kernel=kernel,
            random_state=SEED,
            class_weight="balanced",
        )),
    ])


def run_kfold_tfidf_svm(questions=None, n_folds=N_FOLDS):
    """Run K-fold CV with TF-IDF + SVM, same splits as CCS."""
    if questions is None:
        questions = load_labeled_questions()

    splits = get_kfold_splits(questions, n_folds)
    all_true = []
    all_pred = []
    fold_metrics = []

    for split in splits:
        train_texts = [format_input_text(q) for q in split["train"]]
        val_texts = [format_input_text(q) for q in split["val"]]
        train_labels = [q["bloom_level"] for q in split["train"]]
        val_labels = [q["bloom_level"] for q in split["val"]]

        model = build_tfidf_svm()
        model.fit(train_texts, train_labels)
        preds = model.predict(val_texts).tolist()

        metrics = compute_metrics(val_labels, preds)
        fold_metrics.append(metrics)
        all_true.extend(val_labels)
        all_pred.extend(preds)

        print(f"  Fold {split['fold']}: Acc={metrics['exact_accuracy']:.4f}, "
              f"Adj={metrics['adjacent_accuracy']:.4f}, MAE={metrics['mae']:.4f}")

    # Aggregate
    aggregate = compute_metrics(all_true, all_pred)
    return aggregate, fold_metrics


if __name__ == "__main__":
    print("TF-IDF + SVM Baseline (5-fold CV)")
    print("-" * 40)
    aggregate, fold_metrics = run_kfold_tfidf_svm()
    print_metrics(aggregate, "TF-IDF + SVM (Aggregate)")

    # Per-fold summary
    for key in ["exact_accuracy", "adjacent_accuracy", "mae", "f1_macro"]:
        values = [m[key] for m in fold_metrics]
        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
