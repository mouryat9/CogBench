#!/usr/bin/env python3
"""
CogBench — Export Gold Labels from Annotation

After all annotators have completed their labels and disagreements have been
resolved, this script:

1. Loads Label Studio JSON export
2. Computes majority-vote labels for each question
3. Identifies items needing adjudication (ties, low agreement)
4. Produces the final gold_human_annotated.json
5. Creates train/test split (80/20, stratified by Bloom's level)

Usage:
  python export_gold_labels.py --annotations <export.json> --source <all_collected_questions.json>

After running:
  - data/processed/gold_human_annotated.json — full gold dataset (600 items)
  - data/processed/gold_train.json — 80% for CCS model training (~480 items)
  - data/processed/gold_test.json — 20% held-out test set (~120 items)
  - data/annotations/needs_adjudication.json — items with ties/disagreement
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "annotations")

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}

BLOOM_LEVELS = {
    "L1 Remember": 1, "L2 Understand": 2, "L3 Apply": 3,
    "L4 Analyze": 4, "L5 Evaluate": 5, "L6 Create": 6,
}


def parse_annotations(export_data):
    """Parse Label Studio export into structured annotations."""
    results = {}  # q_id -> list of {annotator, level, confidence, notes}

    for task in export_data:
        q_id = task.get("data", {}).get("question_id", str(task.get("id", "")))
        q_text = task.get("data", {}).get("question_text", "")
        q_type = task.get("data", {}).get("question_type", "unknown")
        subject = task.get("data", {}).get("subject", "unknown")
        source = task.get("data", {}).get("source", "unknown")

        item_annotations = []
        for ann in task.get("annotations", []):
            annotator = str(ann.get("completed_by", ann.get("id", "unknown")))
            level = None
            confidence = None
            notes = ""

            for result in ann.get("result", []):
                name = result.get("from_name")
                choices = result.get("value", {}).get("choices", [])
                text_val = result.get("value", {}).get("text", [""])

                if name == "bloom_level" and choices:
                    level = BLOOM_LEVELS.get(choices[0])
                elif name == "confidence" and choices:
                    confidence = choices[0]
                elif name == "notes":
                    notes = text_val[0] if isinstance(text_val, list) else text_val

            if level is not None:
                item_annotations.append({
                    "annotator_id": annotator,
                    "bloom_level": level,
                    "confidence": confidence,
                    "notes": notes,
                })

        if item_annotations:
            results[q_id] = {
                "question_id": q_id,
                "question_text": q_text,
                "question_type": q_type,
                "subject": subject,
                "source": source,
                "annotations": item_annotations,
            }

    return results


def resolve_labels(annotated_items):
    """
    Resolve final labels using majority vote.
    Returns resolved items and items needing adjudication.
    """
    resolved = []
    needs_adjudication = []

    for q_id, item in annotated_items.items():
        annotations = item["annotations"]
        levels = [a["bloom_level"] for a in annotations]
        level_counts = Counter(levels)
        most_common = level_counts.most_common()

        n_annotators = len(annotations)
        top_count = most_common[0][1]

        # Check if there's a clear majority
        if len(most_common) == 1 or top_count > n_annotators / 2:
            # Clear majority
            majority_level = most_common[0][0]
            agreement = top_count / n_annotators

            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                # Tie — needs adjudication
                resolution = "needs_adjudication"
            else:
                resolution = "majority_vote"
        else:
            # No clear majority
            majority_level = most_common[0][0]
            agreement = top_count / n_annotators
            resolution = "needs_adjudication"

        gold_item = {
            "question_id": q_id,
            "question_text": item["question_text"],
            "question_type": item["question_type"],
            "subject": item["subject"],
            "source": item["source"],
            "gold_label": {
                "bloom_level": majority_level,
                "bloom_name": BLOOM_NAMES.get(majority_level, "Unknown"),
                "annotator_labels": levels,
                "annotator_ids": [a["annotator_id"] for a in annotations],
                "annotator_confidence": [a["confidence"] for a in annotations],
                "annotator_notes": [a["notes"] for a in annotations if a["notes"]],
                "agreement": round(agreement, 3),
                "resolution_method": resolution,
            },
        }

        if resolution == "needs_adjudication":
            needs_adjudication.append(gold_item)
        else:
            resolved.append(gold_item)

    return resolved, needs_adjudication


def stratified_split(items, test_ratio=0.2):
    """
    Split items into train/test sets, stratified by Bloom's level.
    Ensures each level is proportionally represented in both sets.
    """
    by_level = defaultdict(list)
    for item in items:
        level = item["gold_label"]["bloom_level"]
        by_level[level].append(item)

    train = []
    test = []

    for level in sorted(by_level.keys()):
        pool = by_level[level]
        random.shuffle(pool)
        n_test = max(1, round(len(pool) * test_ratio))
        test.extend(pool[:n_test])
        train.extend(pool[n_test:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


def main():
    parser = argparse.ArgumentParser(description="CogBench — Export Gold Labels")
    parser.add_argument("--annotations", required=True,
                        help="Path to Label Studio JSON export")
    args = parser.parse_args()

    print("=" * 60)
    print("CogBench — Export Gold Labels")
    print("=" * 60)

    # Load annotations
    with open(args.annotations) as f:
        export_data = json.load(f)
    annotated = parse_annotations(export_data)
    print(f"\nParsed {len(annotated)} annotated items")

    # Resolve labels
    resolved, needs_adjudication = resolve_labels(annotated)
    print(f"  Resolved (majority vote): {len(resolved)}")
    print(f"  Needs adjudication: {len(needs_adjudication)}")

    # Save adjudication items
    if needs_adjudication:
        adj_path = os.path.join(ANNOTATIONS_DIR, "needs_adjudication.json")
        with open(adj_path, "w") as f:
            json.dump(needs_adjudication, f, indent=2)
        print(f"\nSaved adjudication items: {adj_path}")
        print("  → Review these items manually, then re-run with updated annotations")

    # For now, include adjudicated items with their majority-vote label
    # (can be updated after manual adjudication)
    all_gold = resolved + needs_adjudication

    # Distribution
    print(f"\n--- Gold Label Distribution ---")
    level_dist = Counter(item["gold_label"]["bloom_level"] for item in all_gold)
    for lvl in sorted(level_dist):
        print(f"  L{lvl} {BLOOM_NAMES[lvl]}: {level_dist[lvl]}")

    # Agreement stats
    agreements = [item["gold_label"]["agreement"] for item in all_gold]
    print(f"\n--- Agreement Stats ---")
    print(f"  Mean agreement: {sum(agreements)/len(agreements):.1%}")
    print(f"  Full agreement (100%): {sum(1 for a in agreements if a == 1.0)}")
    print(f"  Majority (>50%): {sum(1 for a in agreements if a > 0.5)}")

    # Save full gold dataset
    gold_path = os.path.join(PROCESSED_DIR, "gold_human_annotated.json")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(gold_path, "w") as f:
        json.dump(all_gold, f, indent=2)
    print(f"\nSaved gold dataset: {gold_path} ({len(all_gold)} items)")

    # Train/test split
    train, test = stratified_split(all_gold, test_ratio=0.2)
    train_path = os.path.join(PROCESSED_DIR, "gold_train.json")
    test_path = os.path.join(PROCESSED_DIR, "gold_test.json")

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test, f, indent=2)

    print(f"\n--- Train/Test Split ---")
    print(f"  Train: {len(train)} items → {train_path}")
    print(f"  Test:  {len(test)} items → {test_path}")

    # Per-level breakdown in each split
    for name, split in [("Train", train), ("Test", test)]:
        dist = Counter(item["gold_label"]["bloom_level"] for item in split)
        print(f"\n  {name} distribution:")
        for lvl in sorted(dist):
            print(f"    L{lvl} {BLOOM_NAMES[lvl]}: {dist[lvl]}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
