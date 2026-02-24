"""
Standardize all collected question sources into a unified JSON schema.

Merges:
  1. Yahya et al. (2012)     — 600 parsed questions (with Bloom's labels)
  2. Mohammed & Omar (2020)  — 141 parsed questions (with Bloom's labels)
  3. OpenStaxQA (HuggingFace) — ~3,000 filtered questions (no Bloom's labels)
  4. OpenStax MCQ (scraped)  — ~300-500 MCQs (no Bloom's labels)

Output schema:
  {
    "question_id":       "cogbench_0001",
    "question_text":     "...",
    "question_type":     "open_ended" | "mcq",
    "choices":           null | ["a", "b", "c", "d"],
    "correct_answer":    null | "...",
    "subject":           "biology" | "physics" | ... | "general",
    "source":            "yahya_2012" | "mohammed_omar" | "openstax_qa" | "openstax_mcq",
    "source_detail":     "openstax_biology_2e_ch3",
    "bloom_level":       4 | null,
    "bloom_name":        "Analyze" | null,
    "bloom_label_source": "original_author" | "taxonomy_mapping" | null
  }
"""

import json
import os
import sys
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input files
YAHYA_FILE = os.path.join(BASE_DIR, "data", "raw", "yahya_2012", "yahya_600_parsed.json")
MOHAMMED_FILE = os.path.join(BASE_DIR, "data", "raw", "mohammed_omar", "mohammed_141_parsed.json")
OPENSTAX_QA_FILE = os.path.join(BASE_DIR, "data", "raw", "openstax", "openstax_qa_filtered.json")
OPENSTAX_MCQ_FILE = os.path.join(BASE_DIR, "data", "raw", "openstax", "openstax_mcq_scraped.json")

# Output
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "all_collected_questions.json")

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}


def load_json(path: str) -> list:
    """Load a JSON file, return empty list if not found."""
    if not os.path.exists(path):
        print(f"  WARNING: Not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded: {path} ({len(data)} items)")
    return data


def normalize_text(text: str) -> str:
    """Normalize whitespace for comparison."""
    return " ".join(text.lower().split())


def deduplicate(questions: list[dict]) -> list[dict]:
    """Remove duplicate questions using normalized text exact match.

    Uses exact matching on normalized (lowercased, whitespace-collapsed) text.
    This is fast (O(n)) and catches the main duplication cases: identical
    questions appearing in both OpenStaxQA and scraped MCQs, or within
    the same source.
    """
    seen = set()
    unique = []

    for q in questions:
        norm = normalize_text(q["question_text"])
        if len(norm) < 10:  # Skip very short entries
            continue
        # Use first 200 chars as key (handles minor trailing differences)
        key = norm[:200]
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique


def process_yahya(data: list) -> list[dict]:
    """Convert Yahya parsed data to standard schema."""
    results = []
    for q in data:
        results.append({
            "question_text": q["question_text"],
            "question_type": q["question_type"],
            "choices": None,
            "correct_answer": None,
            "subject": "general",
            "source": "yahya_2012",
            "source_detail": f"yahya_2012_{q['bloom_name'].lower()}",
            "bloom_level": q["bloom_level"],
            "bloom_name": q["bloom_name"],
            "bloom_label_source": "original_author",
        })
    return results


def process_mohammed(data: list) -> list[dict]:
    """Convert Mohammed & Omar parsed data to standard schema."""
    results = []
    for q in data:
        results.append({
            "question_text": q["question_text"],
            "question_type": q["question_type"],
            "choices": None,
            "correct_answer": None,
            "subject": "general",
            "source": "mohammed_omar",
            "source_detail": f"mohammed_omar_{q['bloom_name'].lower()}",
            "bloom_level": q["bloom_level"],
            "bloom_name": q["bloom_name"],
            "bloom_label_source": "original_author",
        })
    return results


def process_openstax_qa(data: list) -> list[dict]:
    """Convert OpenStaxQA HuggingFace data to standard schema."""
    results = []
    for q in data:
        chapter = q.get("chapter")
        book = q.get("source_book", "unknown")
        results.append({
            "question_text": q["question_text"],
            "question_type": q.get("question_type", "open_ended"),
            "choices": None,
            "correct_answer": None,
            "subject": q.get("subject", "unknown"),
            "source": "openstax_qa",
            "source_detail": f"openstax_{book}_ch{chapter}" if chapter else f"openstax_{book}",
            "bloom_level": None,
            "bloom_name": None,
            "bloom_label_source": None,
        })
    return results


def process_openstax_mcq(data: list) -> list[dict]:
    """Convert scraped OpenStax MCQs to standard schema."""
    results = []
    for q in data:
        chapter = q.get("chapter")
        book = q.get("source_book", "unknown")
        results.append({
            "question_text": q["question_text"],
            "question_type": q.get("question_type", "mcq"),
            "choices": q.get("choices"),
            "correct_answer": q.get("correct_answer"),
            "subject": q.get("subject", "unknown"),
            "source": "openstax_mcq",
            "source_detail": f"openstax_{book}_ch{chapter}_mcq" if chapter else f"openstax_{book}_mcq",
            "bloom_level": None,
            "bloom_name": None,
            "bloom_label_source": None,
        })
    return results


def main():
    print("=" * 60)
    print("CogBench — Standardize All Question Sources")
    print("=" * 60)
    print()

    # Load all sources
    print("Loading sources:")
    yahya = load_json(YAHYA_FILE)
    mohammed = load_json(MOHAMMED_FILE)
    openstax_qa = load_json(OPENSTAX_QA_FILE)
    openstax_mcq = load_json(OPENSTAX_MCQ_FILE)
    print()

    # Convert to standard schema
    all_questions = []
    all_questions.extend(process_yahya(yahya))
    all_questions.extend(process_mohammed(mohammed))
    all_questions.extend(process_openstax_qa(openstax_qa))
    all_questions.extend(process_openstax_mcq(openstax_mcq))

    print(f"Total before deduplication: {len(all_questions)}")

    # Deduplicate
    print("Deduplicating (this may take a moment)...")
    unique = deduplicate(all_questions)
    removed = len(all_questions) - len(unique)
    print(f"Removed {removed} duplicates. Remaining: {len(unique)}")

    # Assign sequential IDs
    for i, q in enumerate(unique, start=1):
        q["question_id"] = f"cogbench_{i:05d}"

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    # Print distribution tables
    print(f"\n{'='*60}")
    print("DISTRIBUTION SUMMARY")
    print(f"{'='*60}")

    # By source
    source_counts = {}
    for q in unique:
        source_counts[q["source"]] = source_counts.get(q["source"], 0) + 1
    print("\nBy source:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")

    # By subject
    subject_counts = {}
    for q in unique:
        subject_counts[q["subject"]] = subject_counts.get(q["subject"], 0) + 1
    print("\nBy subject:")
    for subj, count in sorted(subject_counts.items()):
        print(f"  {subj}: {count}")

    # By question type
    type_counts = {}
    for q in unique:
        type_counts[q["question_type"]] = type_counts.get(q["question_type"], 0) + 1
    print("\nBy question type:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")

    # Bloom's level distribution (only for labeled questions)
    labeled = [q for q in unique if q["bloom_level"] is not None]
    unlabeled = len(unique) - len(labeled)
    print(f"\nBloom's level distribution ({len(labeled)} labeled, {unlabeled} unlabeled):")
    level_counts = {}
    for q in labeled:
        key = f"L{q['bloom_level']} {q['bloom_name']}"
        level_counts[key] = level_counts.get(key, 0) + 1
    for level_num in range(1, 7):
        for key, count in sorted(level_counts.items()):
            if key.startswith(f"L{level_num} "):
                print(f"  {key}: {count}")

    # Sample questions
    print(f"\n{'='*60}")
    print("SAMPLE QUESTIONS (1 per source)")
    print(f"{'='*60}")
    shown_sources = set()
    for q in unique:
        if q["source"] not in shown_sources:
            shown_sources.add(q["source"])
            bloom = f"L{q['bloom_level']} {q['bloom_name']}" if q["bloom_level"] else "unlabeled"
            print(f"\n  Source: {q['source']} | Subject: {q['subject']} | Bloom's: {bloom}")
            text = q["question_text"][:120] + ("..." if len(q["question_text"]) > 120 else "")
            print(f"  Q: {text}")
            if q["choices"]:
                print(f"  Choices: {q['choices'][:3]}{'...' if len(q['choices']) > 3 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
