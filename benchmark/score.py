"""Score generated questions using CCS model.

Reads generation JSON files, scores each question with CCS,
and appends ccs_level, ccs_confidence, and correctness flags.

Usage:
    python benchmark/score.py --input data/processed/generated_questions/gen_qwen2_5_14b_with_definition.json
    python benchmark/score.py --all  # Score all generation files
"""

import os
import sys
import json
import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.config import RESULTS_DIR, CCS_MODE, CCS_CHECKPOINT_DIR


def load_ccs_predictor(mode=CCS_MODE, fold=0):
    """Load the CCS predictor (lazy import to avoid loading model until needed)."""
    from ccs_model.predict import CCSPredictor
    checkpoint = os.path.join(CCS_CHECKPOINT_DIR, f"fold_{fold}", f"best_model_{mode}.pt")
    return CCSPredictor(checkpoint_path=checkpoint, mode=mode)


def score_generation_file(input_path: str, predictor=None, output_path: str = None):
    """Score all generated questions in a file with CCS.

    Adds to each record:
      - ccs_level: predicted Bloom's level (1-6)
      - ccs_confidence: model confidence
      - ccs_raw_score: raw model output
      - is_correct: ccs_level == target level
      - is_adjacent: |ccs_level - target level| <= 1
      - level_error: ccs_level - target level (signed)
    """
    with open(input_path) as f:
        data = json.load(f)

    generations = data.get("generations", [])
    if not generations:
        print(f"  No generations in {input_path}")
        return data

    # Check if already scored
    already_scored = sum(1 for g in generations if "ccs_level" in g)
    if already_scored == len(generations):
        print(f"  Already scored: {input_path} ({already_scored} questions)")
        return data

    # Load CCS if not provided
    if predictor is None:
        predictor = load_ccs_predictor()

    n_scored = 0
    n_skipped = 0

    for g in generations:
        if "ccs_level" in g:
            continue  # Already scored

        question = g.get("generated_question")
        if not question:
            g["ccs_level"] = None
            g["ccs_confidence"] = None
            g["ccs_raw_score"] = None
            g["is_correct"] = False
            g["is_adjacent"] = False
            g["level_error"] = None
            n_skipped += 1
            continue

        score = predictor.predict(question, subject=g.get("subject", "general"))
        target_level = g["level"]

        g["ccs_level"] = score.level
        g["ccs_level_name"] = score.name
        g["ccs_confidence"] = score.confidence
        g["ccs_raw_score"] = score.raw_score
        g["is_correct"] = score.level == target_level
        g["is_adjacent"] = abs(score.level - target_level) <= 1
        g["level_error"] = score.level - target_level

        n_scored += 1

    # Save scored results
    output_path = output_path or input_path.replace(".json", "_scored.json")
    data["scored_timestamp"] = datetime.now().isoformat()
    data["ccs_mode"] = CCS_MODE
    data["n_scored"] = n_scored
    data["n_skipped"] = n_skipped

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Scored {n_scored} questions ({n_skipped} skipped). Saved to: {output_path}")
    return data


def score_all(results_dir=RESULTS_DIR):
    """Score all generation files in the results directory."""
    pattern = os.path.join(results_dir, "gen_*.json")
    files = sorted(glob.glob(pattern))
    # Exclude already-scored files
    files = [f for f in files if "_scored" not in f]

    if not files:
        print(f"No generation files found in {results_dir}")
        return

    print(f"Found {len(files)} generation files to score")

    # Load CCS once
    predictor = load_ccs_predictor()

    for fpath in files:
        print(f"\nScoring: {os.path.basename(fpath)}")
        score_generation_file(fpath, predictor=predictor)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score generated questions with CCS")
    parser.add_argument("--input", type=str, help="Path to generation JSON file")
    parser.add_argument("--all", action="store_true", help="Score all generation files")
    args = parser.parse_args()

    if args.all:
        score_all()
    elif args.input:
        score_generation_file(args.input)
    else:
        print("Specify --input <file> or --all")
