"""Silver-label all 27K questions with Phase 1 CCS model.

Produces a copy of all_collected_questions.json with CCS-predicted Bloom's levels
for all previously unlabeled questions. Original labels (739 from Yahya/Mohammed)
are preserved — only questions with bloom_level=None get silver labels.

Usage:
    python ccs_model/silver_label.py
    python ccs_model/silver_label.py --mode classification
    python ccs_model/silver_label.py --relabel-all  # Override even existing labels
"""

import os
import sys
import json
import time
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccs_model.config import OUTPUT_DIR, BLOOM_LEVELS, PROJECT_ROOT
from ccs_model.predict import CCSPredictor


DATA_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "all_collected_questions.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "all_questions_silver_labeled.json")


def silver_label(mode="classification", relabel_all=False, batch_size=64):
    """Label all questions with CCS model.

    Args:
        mode: 'classification' or 'regression'
        relabel_all: If True, override existing human labels too
        batch_size: Batch size for inference
    """
    # Load data
    print(f"Loading questions from {DATA_FILE}...")
    with open(DATA_FILE) as f:
        questions = json.load(f)
    print(f"  Total questions: {len(questions)}")

    # Count existing labels
    has_label = sum(1 for q in questions if q.get("bloom_level") is not None)
    needs_label = len(questions) - has_label
    print(f"  Already labeled: {has_label}")
    print(f"  Needs labeling: {needs_label}")

    if relabel_all:
        to_label = questions
        print(f"  Relabeling ALL {len(questions)} questions (--relabel-all)")
    else:
        to_label = [q for q in questions if q.get("bloom_level") is None]
        print(f"  Labeling {len(to_label)} unlabeled questions")

    if not to_label:
        print("  Nothing to label!")
        return

    # Load CCS model
    print(f"\nLoading CCS model (mode={mode})...")
    predictor = CCSPredictor(mode=mode)

    # Batch predict with progress tracking
    print(f"\nRunning inference (batch_size={batch_size})...")
    t_start = time.time()
    scores = []
    n_total = len(to_label)

    for i in range(0, n_total, batch_size):
        batch = to_label[i:i + batch_size]
        batch_scores = predictor.predict_batch(batch, batch_size=batch_size)
        scores.extend(batch_scores)

        done = min(i + batch_size, n_total)
        if done % 1000 < batch_size or done == n_total:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else 0
            print(f"  {done}/{n_total} ({done/n_total*100:.0f}%) | "
                  f"{rate:.0f} q/s | ETA: {eta:.0f}s")

    elapsed = time.time() - t_start
    rate = len(to_label) / elapsed
    print(f"  Done: {len(to_label)} questions in {elapsed:.1f}s ({rate:.0f} q/s)")

    # Ensure all bloom_level values are plain Python int (not numpy int64)
    for q in questions:
        if q.get("bloom_level") is not None:
            q["bloom_level"] = int(q["bloom_level"])

    # Apply silver labels
    if relabel_all:
        for q, score in zip(questions, scores):
            q["silver_level"] = score.level
            q["silver_name"] = score.name
            q["silver_confidence"] = round(score.confidence, 4)
            q["silver_raw_score"] = round(score.raw_score, 4)
            # Don't overwrite original bloom_level — store separately
            if q.get("bloom_level") is None:
                q["bloom_level"] = score.level
                q["bloom_name"] = score.name
                q["bloom_label_source"] = "ccs_phase1_silver"
    else:
        # Build index for unlabeled questions
        unlabeled_indices = [i for i, q in enumerate(questions) if q.get("bloom_level") is None]
        for idx, score in zip(unlabeled_indices, scores):
            questions[idx]["bloom_level"] = score.level
            questions[idx]["bloom_name"] = score.name
            questions[idx]["bloom_label_source"] = "ccs_phase1_silver"
            questions[idx]["silver_confidence"] = round(score.confidence, 4)
            questions[idx]["silver_raw_score"] = round(score.raw_score, 4)

    # Summary statistics
    print(f"\n{'=' * 60}")
    print(f"  SILVER LABELING SUMMARY")
    print(f"{'=' * 60}")

    # Level distribution
    level_counts = Counter(q["bloom_level"] for q in questions)
    print(f"\n  Level Distribution (all {len(questions)} questions):")
    print(f"  {'Level':<15s} {'Count':>7s} {'%':>7s}")
    print(f"  {'─' * 30}")
    for lv in sorted(BLOOM_LEVELS.keys()):
        count = level_counts.get(lv, 0)
        pct = count / len(questions) * 100
        print(f"  L{lv} {BLOOM_LEVELS[lv]:<10s} {count:>7d} {pct:>6.1f}%")

    # By source
    print(f"\n  Level Distribution by Source:")
    sources = sorted(set(q.get("source", "unknown") for q in questions))
    print(f"  {'Source':<20s}", end="")
    for lv in sorted(BLOOM_LEVELS.keys()):
        print(f" {'L'+str(lv):>5s}", end="")
    print(f" {'Total':>7s}")
    print(f"  {'─' * 55}")
    for source in sources:
        source_qs = [q for q in questions if q.get("source") == source]
        print(f"  {source:<20s}", end="")
        for lv in sorted(BLOOM_LEVELS.keys()):
            count = sum(1 for q in source_qs if q.get("bloom_level") == lv)
            print(f" {count:>5d}", end="")
        print(f" {len(source_qs):>7d}")

    # By subject
    print(f"\n  Level Distribution by Subject:")
    subjects = sorted(set(q.get("subject", "unknown") for q in questions))
    print(f"  {'Subject':<20s}", end="")
    for lv in sorted(BLOOM_LEVELS.keys()):
        print(f" {'L'+str(lv):>5s}", end="")
    print(f" {'Total':>7s}")
    print(f"  {'─' * 55}")
    for subject in subjects:
        sub_qs = [q for q in questions if q.get("subject") == subject]
        print(f"  {subject:<20s}", end="")
        for lv in sorted(BLOOM_LEVELS.keys()):
            count = sum(1 for q in sub_qs if q.get("bloom_level") == lv)
            print(f" {count:>5d}", end="")
        print(f" {len(sub_qs):>7d}")

    # Confidence stats
    silver_confs = [q.get("silver_confidence", 0) for q in questions
                    if q.get("bloom_label_source") == "ccs_phase1_silver"]
    if silver_confs:
        import numpy as np
        confs = np.array(silver_confs)
        print(f"\n  Silver Label Confidence:")
        print(f"  Mean: {confs.mean():.3f}, Median: {np.median(confs):.3f}, "
              f"Min: {confs.min():.3f}, Max: {confs.max():.3f}")
        print(f"  >0.9: {(confs > 0.9).sum()} ({(confs > 0.9).mean():.1%})")
        print(f"  >0.7: {(confs > 0.7).sum()} ({(confs > 0.7).mean():.1%})")
        print(f"  <0.5: {(confs < 0.5).sum()} ({(confs < 0.5).mean():.1%})")

    # Save
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "ccs_mode": mode,
            "ccs_phase": "phase1",
            "n_total": len(questions),
            "n_original_labels": has_label,
            "n_silver_labels": needs_label if not relabel_all else len(questions),
            "relabel_all": relabel_all,
        },
        "questions": questions,
    }

    # Convert numpy types for JSON serialization
    def _convert(obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            converted = _convert(obj)
            if converted is not obj:
                return converted
            return super().default(obj)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved to: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

    return questions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Silver-label questions with CCS")
    parser.add_argument("--mode", default="classification", choices=["classification", "regression"])
    parser.add_argument("--relabel-all", action="store_true",
                        help="Override existing human labels too")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    silver_label(mode=args.mode, relabel_all=args.relabel_all, batch_size=args.batch_size)
