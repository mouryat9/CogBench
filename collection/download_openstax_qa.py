"""
Download the OpenStaxQA dataset from HuggingFace and filter to target textbooks.

Source: pranav-gupta/openstax-qa (arXiv:2510.06239, October 2025)
  - 21,500 problem-solution pairs from 43 OpenStax textbooks
  - CC-BY 4.0 licensed
  - Primarily free-response / critical-thinking questions

We filter to ALL English-language textbooks for maximum cross-domain coverage.

Excludes: foreign-language editions (Spanish, Polish), exact content
duplicates (chemistry-atoms-first-2e, college-physics-ap-courses-2e,
college-algebra-corequisite-support-2e, statistics duplicate).
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw", "openstax")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "openstax_qa_filtered.json")

# ALL English-language HuggingFace books → subject
# Excluded: chemistry-atoms-first-2e (dup of chemistry-2e),
#   college-physics-ap-courses-2e (dup of college-physics-2e),
#   college-algebra-corequisite-support-2e (dup of college-algebra-2e),
#   all Spanish (cálculo, física, química, precálculo) and Polish editions
TARGET_BOOKS = {
    # Biology
    "microbiology":                      "biology",
    # Chemistry
    "chemistry-2e":                      "chemistry",
    # Physics
    "college-physics-2e":                "physics",
    "university-physics-volume-1":       "physics",
    "university-physics-volume-2":       "physics",
    "university-physics-volume-3":       "physics",
    # Mathematics
    "calculus-volume-1":                 "mathematics",
    "calculus-volume-2":                 "mathematics",
    "calculus-volume-3":                 "mathematics",
    "college-algebra-2e":                "mathematics",
    "algebra-and-trigonometry-2e":       "mathematics",
    "precalculus-2e":                    "mathematics",
    "prealgebra-2e":                     "mathematics",
    "elementary-algebra-2e":             "mathematics",
    "intermediate-algebra-2e":           "mathematics",
    # Economics
    "principles-economics-3e":           "economics",
    "principles-macroeconomics-3e":      "economics",
    "principles-microeconomics-3e":      "economics",
    # Political Science
    "american-government-3e":            "political_science",
    # Business
    "business-ethics":                   "business",
    "business-law-i-essentials":         "business",
    "introduction-intellectual-property": "business",
    "principles-financial-accounting":   "business",
    "principles-managerial-accounting":  "business",
    # Sociology
    "introduction-sociology-3e":         "sociology",
    # History
    "us-history":                        "history",
}


def match_subject(book_name: str) -> str | None:
    """Match a HuggingFace book name to our target subject via exact lookup."""
    return TARGET_BOOKS.get(book_name)


def main():
    print("=" * 60)
    print("CogBench — Download OpenStaxQA from HuggingFace")
    print("=" * 60)
    print()

    # Import here so the script fails fast with a clear message
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        return 1

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            existing = json.load(f)
        print(f"Already exists: {OUTPUT_FILE} ({len(existing)} questions)")
        print("Delete it to re-download.")
        return 0

    print("Loading dataset from HuggingFace (pranav-gupta/openstax-qa)...")
    print("  This may take a minute on first download.\n")

    ds = load_dataset("pranav-gupta/openstax-qa")

    # Explore available splits and book names
    all_questions = []
    for split_name in ds:
        split = ds[split_name]
        print(f"Split '{split_name}': {len(split)} rows")

        # Get column names
        if not all_questions:
            print(f"  Columns: {split.column_names}")

        for row in split:
            all_questions.append(row)

    print(f"\nTotal rows across all splits: {len(all_questions)}")

    # Discover unique book names
    book_field = "book" if "book" in all_questions[0] else None
    if not book_field:
        # Try to find the book field
        for key in all_questions[0]:
            sample_val = str(all_questions[0][key])
            if any(kw in sample_val.lower() for kw in ["biology", "physics"]):
                book_field = key
                break

    if book_field:
        unique_books = sorted(set(str(q[book_field]) for q in all_questions))
        print(f"\nUnique books (field: '{book_field}'):")
        for b in unique_books:
            count = sum(1 for q in all_questions if str(q[book_field]) == b)
            subject = match_subject(b)
            marker = " ← TARGET" if subject else ""
            print(f"  {b}: {count} questions{marker}")

    # Filter to target textbooks
    filtered = []
    for q in all_questions:
        book_name = str(q.get(book_field, "")) if book_field else ""
        subject = match_subject(book_name)
        if subject:
            # Use cleaned text if available, fall back to raw
            question_text = q.get("problem_cleaned") or q.get("problem", "")
            solution_text = q.get("solution_cleaned") or q.get("solution", "")

            if not question_text.strip():
                continue

            filtered.append({
                "question_text": question_text.strip(),
                "solution_text": solution_text.strip() if solution_text else None,
                "question_type": "open_ended",
                "subject": subject,
                "source_book": book_name,
                "chapter": q.get("chapter_number"),
                "source_id": q.get("id"),
            })

    print(f"\nFiltered to target textbooks: {len(filtered)} questions")

    # Distribution by subject
    subject_counts = {}
    for q in filtered:
        subject_counts[q["subject"]] = subject_counts.get(q["subject"], 0) + 1
    print("\nDistribution by subject:")
    for subj, count in sorted(subject_counts.items()):
        print(f"  {subj}: {count}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
