#!/usr/bin/env python3
"""
CogBench — Sample Questions for Human Annotation (Split-Overlap Strategy)

With 5 annotators targeting ~300 questions each (~8-10 hours per person),
we use a split-overlap design:

  CORE SET (150 questions):  ALL 5 annotators label these.
    → Used for computing inter-annotator agreement (Fleiss' kappa).
    → 150 items × 5 raters = very stable IAA estimate.

  EXTENDED SET (250 questions):  3 annotators each, round-robin assigned.
    → Expands the gold dataset without overloading anyone.
    → Every question still gets 3+ labels for majority vote.

  TOTAL: 400 questions, ~300 per annotator, ~8-10 hours each.
  (320 train / 80 test after 80/20 split for CCS model.)

Also produces:
  - training_set.json — 36 pre-labeled questions for the calibration session.
  - Gold-check questions hidden in both core and extended sets.

Math:
  Core:     150 × 5 annotators = 750 annotation slots
  Extended: 250 × 3 annotators = 750 annotation slots
  Total:    1500 slots / 5 annotators = 300 per person ✓
"""

import json
import os
import random
from collections import defaultdict

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_QUESTIONS_FILE = os.path.join(BASE_DIR, "data", "processed", "all_collected_questions.json")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "annotations")
ANNOTATION_SAMPLE_FILE = os.path.join(ANNOTATIONS_DIR, "annotation_sample.json")
TRAINING_SET_FILE = os.path.join(ANNOTATIONS_DIR, "training_set.json")

NUM_ANNOTATORS = 5
CORE_COUNT = 150        # all 5 annotators label these (IAA measurement)
EXTENDED_COUNT = 250    # 3 annotators each (round-robin assignment)
TOTAL_SAMPLE = CORE_COUNT + EXTENDED_COUNT  # 400
GOLD_CHECK_COUNT = 20   # hidden gold questions (split across core + extended)
TRAINING_COUNT = 36     # pre-labeled questions for calibration (6 per level)
MIN_PER_SUBJECT = 8     # minimum questions per subject


def load_questions(path):
    with open(path) as f:
        return json.load(f)


def split_labeled_unlabeled(questions):
    """Separate pre-labeled (Yahya + Mohammed) from unlabeled (OpenStax)."""
    labeled = [q for q in questions if q.get("bloom_level") is not None]
    unlabeled = [q for q in questions if q.get("bloom_level") is None]
    return labeled, unlabeled


def create_training_set(labeled_questions, count=TRAINING_COUNT):
    """Sample balanced training set from pre-labeled questions (~13 per level)."""
    by_level = defaultdict(list)
    for q in labeled_questions:
        by_level[q["bloom_level"]].append(q)

    per_level = count // 6
    remainder = count - (per_level * 6)

    training = []
    for level in sorted(by_level.keys()):
        pool = by_level[level]
        n = per_level + (1 if level <= remainder else 0)
        sampled = random.sample(pool, min(n, len(pool)))
        training.extend(sampled)

    random.shuffle(training)
    return training


def create_gold_checks(labeled_questions, exclude_ids, count=GOLD_CHECK_COUNT):
    """Sample gold-check questions from pre-labeled set, balanced across levels."""
    eligible = [q for q in labeled_questions if q["question_id"] not in exclude_ids]

    by_level = defaultdict(list)
    for q in eligible:
        by_level[q["bloom_level"]].append(q)

    per_level = count // 6
    remainder = count - (per_level * 6)

    gold = []
    for level in sorted(by_level.keys()):
        pool = by_level[level]
        n = per_level + (1 if level <= remainder else 0)
        sampled = random.sample(pool, min(n, len(pool)))
        gold.extend(sampled)

    return gold


def stratified_sample(unlabeled_questions, target_count, min_per_subject=MIN_PER_SUBJECT):
    """Stratified sample ensuring subject and type coverage."""
    by_subject = defaultdict(list)
    for q in unlabeled_questions:
        by_subject[q["subject"]].append(q)

    subjects = sorted(by_subject.keys())

    # Phase 1: Guarantee minimum per subject
    sampled = []
    sampled_ids = set()

    for subj in subjects:
        pool = by_subject[subj]
        n = min(min_per_subject, len(pool))
        chosen = random.sample(pool, n)
        for q in chosen:
            sampled.append(q)
            sampled_ids.add(q["question_id"])

    # Phase 2: Fill remaining proportionally
    remaining = target_count - len(sampled)
    if remaining > 0:
        leftover_pool = [q for q in unlabeled_questions if q["question_id"] not in sampled_ids]
        by_subject_leftover = defaultdict(list)
        for q in leftover_pool:
            by_subject_leftover[q["subject"]].append(q)

        total_leftover = len(leftover_pool)
        for subj in sorted(by_subject_leftover.keys()):
            pool = by_subject_leftover[subj]
            proportion = len(pool) / total_leftover
            n = round(proportion * remaining)
            n = min(n, len(pool))
            chosen = random.sample(pool, n)
            for q in chosen:
                if q["question_id"] not in sampled_ids:
                    sampled.append(q)
                    sampled_ids.add(q["question_id"])

    # Trim or pad
    if len(sampled) > target_count:
        sampled = sampled[:target_count]
    elif len(sampled) < target_count:
        leftover = [q for q in unlabeled_questions if q["question_id"] not in sampled_ids]
        extra = random.sample(leftover, min(target_count - len(sampled), len(leftover)))
        sampled.extend(extra)

    return sampled


def assign_extended_round_robin(extended_questions, num_annotators=NUM_ANNOTATORS, raters_per_item=3):
    """
    Assign 3 annotators per extended question using round-robin.
    Returns dict: annotator_index (0-4) -> list of question_ids.

    Uses a balanced assignment: cycle through all C(5,3)=10 possible
    3-annotator combinations to ensure even distribution.
    """
    from itertools import combinations

    combos = list(combinations(range(num_annotators), raters_per_item))
    random.shuffle(combos)

    assignments = {i: [] for i in range(num_annotators)}
    question_assignments = {}  # q_id -> list of annotator indices

    for idx, q in enumerate(extended_questions):
        combo = combos[idx % len(combos)]
        question_assignments[q["question_id"]] = list(combo)
        for annotator_idx in combo:
            assignments[annotator_idx].append(q["question_id"])

    return assignments, question_assignments


def format_for_label_studio(questions, is_gold=False):
    """Format questions for Label Studio JSON import."""
    tasks = []
    for q in questions:
        task = {
            "data": {
                "question_id": q["question_id"],
                "question_text": q["question_text"],
                "question_type": q.get("question_type", "unknown"),
                "subject": q.get("subject", "unknown"),
                "source": q.get("source", "unknown"),
            }
        }
        if is_gold and q.get("bloom_level") is not None:
            task["data"]["_gold_bloom_level"] = q["bloom_level"]
            task["data"]["_gold_bloom_name"] = q.get("bloom_name", "")
            task["data"]["_is_gold_check"] = True
        tasks.append(task)
    return tasks


def main():
    print("=" * 60)
    print("CogBench — Sample Questions for Annotation (Split-Overlap)")
    print("=" * 60)

    questions = load_questions(ALL_QUESTIONS_FILE)
    print(f"\nLoaded {len(questions)} total questions")

    labeled, unlabeled = split_labeled_unlabeled(questions)
    print(f"  Pre-labeled (Yahya + Mohammed): {len(labeled)}")
    print(f"  Unlabeled (OpenStax): {len(unlabeled)}")

    # ─── 1. Training set (36 questions from pre-labeled) ───
    print(f"\n{'─'*50}")
    print(f"TRAINING SET ({TRAINING_COUNT} questions)")
    print(f"{'─'*50}")
    training = create_training_set(labeled, TRAINING_COUNT)
    training_ids = {q["question_id"] for q in training}
    print(f"  Sampled: {len(training)} training questions")
    level_dist = defaultdict(int)
    for q in training:
        level_dist[q["bloom_level"]] += 1
    for lvl in sorted(level_dist):
        print(f"    L{lvl}: {level_dist[lvl]}")

    # ─── 2. Gold-check questions (20 from pre-labeled) ───
    print(f"\n{'─'*50}")
    print(f"GOLD CHECK QUESTIONS ({GOLD_CHECK_COUNT})")
    print(f"{'─'*50}")
    gold_checks = create_gold_checks(labeled, training_ids, GOLD_CHECK_COUNT)
    gold_ids = {q["question_id"] for q in gold_checks}
    print(f"  Sampled: {len(gold_checks)} gold-check questions")

    # ─── 3. Sample annotation questions (380 unlabeled + 20 gold = 400) ───
    unlabeled_target = TOTAL_SAMPLE - len(gold_checks)
    print(f"\n{'─'*50}")
    print(f"ANNOTATION SAMPLE ({TOTAL_SAMPLE} total)")
    print(f"{'─'*50}")
    print(f"  Sampling {unlabeled_target} from unlabeled + {len(gold_checks)} gold checks")

    annotation_unlabeled = stratified_sample(unlabeled, unlabeled_target)
    all_annotation = annotation_unlabeled + gold_checks
    random.shuffle(all_annotation)
    print(f"  Total: {len(all_annotation)} questions")

    # ─── 4. Split into CORE (150) and EXTENDED (250) ───
    print(f"\n{'─'*50}")
    print(f"SPLIT-OVERLAP ASSIGNMENT")
    print(f"{'─'*50}")

    # Shuffle and split
    random.shuffle(all_annotation)
    core_questions = all_annotation[:CORE_COUNT]
    extended_questions = all_annotation[CORE_COUNT:]

    # Count gold checks in each set
    core_gold = sum(1 for q in core_questions if q["question_id"] in gold_ids)
    ext_gold = sum(1 for q in extended_questions if q["question_id"] in gold_ids)

    print(f"\n  CORE SET: {len(core_questions)} questions")
    print(f"    → ALL {NUM_ANNOTATORS} annotators label these (for IAA)")
    print(f"    → Gold checks hidden: {core_gold}")

    print(f"\n  EXTENDED SET: {len(extended_questions)} questions")
    print(f"    → 3 of {NUM_ANNOTATORS} annotators each (round-robin)")
    print(f"    → Gold checks hidden: {ext_gold}")

    # Assign extended questions to annotators
    ext_assignments, ext_question_map = assign_extended_round_robin(extended_questions)

    print(f"\n  Per-annotator workload:")
    for i in range(NUM_ANNOTATORS):
        total = CORE_COUNT + len(ext_assignments[i])
        hours_low = total * 2 / 60
        hours_high = total * 3 / 60
        print(f"    Annotator {i+1}: {CORE_COUNT} core + {len(ext_assignments[i])} extended = {total} total (~{hours_low:.0f}-{hours_high:.0f} hrs)")

    # ─── 5. Print distributions ───
    print(f"\n{'─'*50}")
    print(f"DISTRIBUTIONS")
    print(f"{'─'*50}")
    subject_dist = defaultdict(int)
    type_dist = defaultdict(int)
    source_dist = defaultdict(int)
    for q in all_annotation:
        subject_dist[q["subject"]] += 1
        type_dist[q.get("question_type", "unknown")] += 1
        source_dist[q.get("source", "unknown")] += 1

    print(f"\n  By subject ({len(subject_dist)} subjects):")
    for subj in sorted(subject_dist, key=lambda s: -subject_dist[s]):
        print(f"    {subj}: {subject_dist[subj]}")
    print(f"\n  By question type:")
    for qt in sorted(type_dist):
        print(f"    {qt}: {type_dist[qt]}")
    print(f"\n  By source:")
    for src in sorted(source_dist):
        print(f"    {src}: {source_dist[src]}")

    # ─── 6. Save everything ───
    print(f"\n{'─'*50}")
    print(f"SAVING FILES")
    print(f"{'─'*50}")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    import_dir = os.path.join(ANNOTATIONS_DIR, "label_studio_import")
    os.makedirs(import_dir, exist_ok=True)

    # Training set (with gold labels visible)
    training_tasks = format_for_label_studio(training, is_gold=True)
    with open(TRAINING_SET_FILE, "w") as f:
        json.dump(training_tasks, f, indent=2)
    print(f"  Training set: {TRAINING_SET_FILE} ({len(training_tasks)} tasks)")

    # Core set (all annotators)
    core_tasks = format_for_label_studio(core_questions)
    with open(os.path.join(import_dir, "core_tasks.json"), "w") as f:
        json.dump(core_tasks, f, indent=2)
    print(f"  Core tasks: label_studio_import/core_tasks.json ({len(core_tasks)} tasks)")

    # Extended set (full list — Label Studio will handle distribution)
    extended_tasks = format_for_label_studio(extended_questions)
    with open(os.path.join(import_dir, "extended_tasks.json"), "w") as f:
        json.dump(extended_tasks, f, indent=2)
    print(f"  Extended tasks: label_studio_import/extended_tasks.json ({len(extended_tasks)} tasks)")

    # Per-annotator assignment lists (for reference / manual assignment)
    annotator_assignments = {}
    for i in range(NUM_ANNOTATORS):
        name = f"annotator_{i+1}"
        core_ids = [q["question_id"] for q in core_questions]
        ext_ids = ext_assignments[i]
        annotator_assignments[name] = {
            "core_question_ids": core_ids,
            "extended_question_ids": ext_ids,
            "total": len(core_ids) + len(ext_ids),
        }
    with open(os.path.join(import_dir, "annotator_assignments.json"), "w") as f:
        json.dump(annotator_assignments, f, indent=2)
    print(f"  Assignments: label_studio_import/annotator_assignments.json")

    # Gold-check mapping (admin only)
    gold_mapping = {}
    for q in gold_checks:
        gold_mapping[q["question_id"]] = {
            "bloom_level": q["bloom_level"],
            "bloom_name": q.get("bloom_name", ""),
        }
    with open(os.path.join(import_dir, "gold_check_mapping.json"), "w") as f:
        json.dump(gold_mapping, f, indent=2)
    print(f"  Gold mapping: label_studio_import/gold_check_mapping.json (ADMIN ONLY)")

    # Full annotation metadata
    full_output = {
        "metadata": {
            "total_questions": TOTAL_SAMPLE,
            "core_count": CORE_COUNT,
            "extended_count": EXTENDED_COUNT,
            "gold_check_count": len(gold_checks),
            "num_annotators": NUM_ANNOTATORS,
            "annotations_per_core": NUM_ANNOTATORS,
            "annotations_per_extended": 3,
            "per_annotator_target": CORE_COUNT + EXTENDED_COUNT * 3 // NUM_ANNOTATORS,
            "sampling_seed": 42,
        },
        "core_question_ids": [q["question_id"] for q in core_questions],
        "extended_question_ids": [q["question_id"] for q in extended_questions],
        "extended_assignments": {f"annotator_{i+1}": ext_assignments[i] for i in range(NUM_ANNOTATORS)},
        "gold_check_mapping": gold_mapping,
    }
    with open(ANNOTATION_SAMPLE_FILE, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"  Full metadata: {ANNOTATION_SAMPLE_FILE}")

    # Also save the combined task list for Label Studio import (all 400)
    all_tasks = format_for_label_studio(all_annotation)
    with open(os.path.join(import_dir, "all_annotation_tasks.json"), "w") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"  All tasks: label_studio_import/all_annotation_tasks.json ({len(all_tasks)} tasks)")

    print(f"\n{'=' * 60}")
    print("DONE — Split-overlap annotation sample ready")
    print(f"{'=' * 60}")
    print(f"""
Summary:
  400 total questions (150 core + 250 extended)
  ~300 per annotator × 5 annotators
  ~8-10 hours per person
  {len(gold_checks)} hidden gold checks for quality monitoring
  {len(training)} training questions for calibration session
""")


if __name__ == "__main__":
    main()
