"""
Parse downloaded DOCX datasets into structured JSON.

Reads the Yahya (2012) and Mohammed & Omar (2020) DOCX files, extracts
questions grouped by Bloom's level headings, and maps from the Original
Bloom's Taxonomy (1956) to the Revised Bloom's Taxonomy (2001).

Critical mapping note:
  Original L5 "Synthesis" (making new things)  → Revised L6 "Create"
  Original L6 "Evaluation" (judging/defending)  → Revised L5 "Evaluate"
  The other four levels (L1-L4) map 1:1 with just a name change.
"""

import json
import os
import sys
from docx import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Original Bloom's (1956) → Revised Bloom's (2001)
# Key: The top two levels SWAP — Synthesis→Create(6), Evaluation→Evaluate(5)
ORIGINAL_TO_REVISED = {
    "knowledge":     {"name": "Remember",    "level": 1},
    "comprehension": {"name": "Understand",  "level": 2},
    "application":   {"name": "Apply",       "level": 3},
    "analysis":      {"name": "Analyze",     "level": 4},
    "synthesis":     {"name": "Create",      "level": 6},  # NOT 5
    "evaluation":    {"name": "Evaluate",    "level": 5},  # NOT 6
}

VALID_LEVELS = set(ORIGINAL_TO_REVISED.keys())


def parse_docx(docx_path: str) -> list[dict]:
    """
    Parse a Bloom's Taxonomy DOCX file into a list of question dicts.

    The DOCX format is: bold Normal-style paragraphs are Bloom's level
    headers, and List Paragraph entries below them are the questions.
    """
    doc = Document(docx_path)
    questions = []
    current_level_original = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        is_bold = any(r.bold for r in para.runs) if para.runs else False

        # Detect level headers: bold, Normal style, text matches a known level
        if is_bold and para.style.name == "Normal":
            level_key = text.lower()
            if level_key in VALID_LEVELS:
                current_level_original = level_key
            # Skip non-level bold text (e.g., Mohammed's opening note)
            continue

        # Collect questions under the current level
        if para.style.name == "List Paragraph" and current_level_original:
            revised = ORIGINAL_TO_REVISED[current_level_original]
            questions.append({
                "question_text": text,
                "question_type": "open_ended",
                "bloom_level_original": current_level_original.capitalize(),
                "bloom_level": revised["level"],
                "bloom_name": revised["name"],
            })

    return questions


def main():
    datasets = [
        {
            "name": "Yahya et al. (2012)",
            "input": os.path.join(BASE_DIR, "data", "raw", "yahya_2012", "s2_yahya.docx"),
            "output": os.path.join(BASE_DIR, "data", "raw", "yahya_2012", "yahya_600_parsed.json"),
            "expected": 600,
        },
        {
            "name": "Mohammed & Omar (2020)",
            "input": os.path.join(BASE_DIR, "data", "raw", "mohammed_omar", "s1_collected.docx"),
            "output": os.path.join(BASE_DIR, "data", "raw", "mohammed_omar", "mohammed_141_parsed.json"),
            "expected": 141,
        },
    ]

    print("=" * 60)
    print("CogBench — Parse DOCX Bloom's Taxonomy Datasets")
    print("=" * 60)
    print()

    for ds in datasets:
        print(f"Parsing: {ds['name']}")
        print(f"  Input:  {ds['input']}")

        if not os.path.exists(ds["input"]):
            print(f"  ERROR: File not found. Run download_datasets.py first.")
            continue

        questions = parse_docx(ds["input"])
        print(f"  Parsed: {len(questions)} questions (expected {ds['expected']})")

        if len(questions) != ds["expected"]:
            print(f"  WARNING: Count mismatch!")

        # Distribution
        level_counts = {}
        for q in questions:
            key = f"L{q['bloom_level']} {q['bloom_name']}"
            level_counts[key] = level_counts.get(key, 0) + 1

        print(f"  Distribution (Revised Bloom's):")
        for level_num in range(1, 7):
            for key, count in sorted(level_counts.items()):
                if key.startswith(f"L{level_num} "):
                    print(f"    {key}: {count}")

        # Save
        with open(ds["output"], "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"  Saved:  {ds['output']}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
