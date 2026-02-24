"""
Scrape questions from OpenStax textbook websites using server-side rendered HTML.

OpenStax pages are server-side rendered — the full content is in the initial
HTML response. No browser needed. This is fast (~0.5s/page) and reliable.

Supports resume — saves progress after each page.

Two exercise formats exist across books:
  - data-type="exercise"           (most books)
  - data-type="injected-exercise"  (newer books: CS, contemporary-math)
Both use .os-problem-container for the question body.
"""

import json
import os
import sys
import time
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "openstax", "openstax_mcq_scraped.json")
PROGRESS_FILE = os.path.join(BASE_DIR, "data", "raw", "openstax", "scrape_progress.json")

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"}

# Each book: slug, subject, list of page suffixes to try, number of chapters
BOOKS = [
    # ── Biology ──
    {"slug": "biology-2e", "subject": "biology", "page_suffixes": ["review-questions", "critical-thinking-questions"], "chapters": 47},
    {"slug": "anatomy-and-physiology-2e", "subject": "biology", "page_suffixes": ["review-questions", "critical-thinking-questions"], "chapters": 28},
    {"slug": "concepts-biology", "subject": "biology", "page_suffixes": ["review-questions", "critical-thinking-questions"], "chapters": 21},
    # ── Chemistry ──
    {"slug": "chemistry-2e", "subject": "chemistry", "page_suffixes": ["exercises"], "chapters": 21},
    # ── Physics ──
    {"slug": "college-physics-2e", "subject": "physics", "page_suffixes": ["conceptual-questions"], "chapters": 35},
    {"slug": "physics", "subject": "physics", "page_suffixes": ["concept-items", "critical-thinking-items", "multiple-choice", "short-answer"], "chapters": 23},
    # ── Astronomy ──
    {"slug": "astronomy-2e", "subject": "astronomy", "page_suffixes": ["review-questions", "thought-questions"], "chapters": 30},
    # ── Mathematics ──
    {"slug": "introductory-statistics-2e", "subject": "mathematics", "page_suffixes": ["chapter-review", "practice"], "chapters": 13},
    {"slug": "contemporary-mathematics", "subject": "mathematics", "page_suffixes": ["chapter-review", "chapter-test"], "chapters": 14},
    {"slug": "calculus-volume-1", "subject": "mathematics", "page_suffixes": ["review-exercises"], "chapters": 6},
    {"slug": "calculus-volume-2", "subject": "mathematics", "page_suffixes": ["review-exercises"], "chapters": 7},
    {"slug": "calculus-volume-3", "subject": "mathematics", "page_suffixes": ["review-exercises"], "chapters": 7},
    # ── Computer Science ──
    {"slug": "introduction-computer-science", "subject": "computer_science", "page_suffixes": ["review-questions", "conceptual-questions", "thought-provokers"], "chapters": 14},
    {"slug": "foundations-information-systems", "subject": "computer_science", "page_suffixes": ["review-questions", "check-your-understanding-questions"], "chapters": 12},
    {"slug": "principles-data-science", "subject": "computer_science", "page_suffixes": ["chapter-review", "critical-thinking"], "chapters": 11},
    # ── Economics ──
    {"slug": "principles-economics-3e", "subject": "economics", "page_suffixes": ["review-questions", "critical-thinking-questions", "self-check-questions"], "chapters": 35},
    # ── Political Science ──
    {"slug": "american-government-4e", "subject": "political_science", "page_suffixes": ["review-questions", "critical-thinking-questions"], "chapters": 17},
    {"slug": "introduction-political-science", "subject": "political_science", "page_suffixes": ["review-questions"], "chapters": 16},
    # ── Psychology ──
    {"slug": "psychology-2e", "subject": "psychology", "page_suffixes": ["review-questions", "critical-thinking-questions"], "chapters": 16},
    {"slug": "lifespan-development", "subject": "psychology", "page_suffixes": ["review-questions", "check-your-understanding-questions", "essay-questions"], "chapters": 17},
    {"slug": "introduction-behavioral-neuroscience", "subject": "psychology", "page_suffixes": ["multiple-choice", "fill-in-the-blank"], "chapters": 20},
    # ── Sociology / Anthropology ──
    {"slug": "introduction-sociology-3e", "subject": "sociology", "page_suffixes": ["section-quiz", "short-answer"], "chapters": 22},
    {"slug": "introduction-anthropology", "subject": "anthropology", "page_suffixes": ["critical-thinking-questions"], "chapters": 20},
    # ── History ──
    {"slug": "us-history", "subject": "history", "page_suffixes": ["review-questions", "critical-thinking-questions"], "chapters": 33},
    {"slug": "world-history-volume-1", "subject": "history", "page_suffixes": ["assessments"], "chapters": 17},
    {"slug": "world-history-volume-2", "subject": "history", "page_suffixes": ["assessments"], "chapters": 15},
    # ── Philosophy ──
    {"slug": "introduction-philosophy", "subject": "philosophy", "page_suffixes": ["review-questions"], "chapters": 12},
    # ── Business ──
    {"slug": "business-ethics", "subject": "business", "page_suffixes": ["assessment-questions"], "chapters": 12},
    {"slug": "business-law-i-essentials", "subject": "business", "page_suffixes": ["assessment-questions"], "chapters": 14},
    {"slug": "entrepreneurship", "subject": "business", "page_suffixes": ["review-questions", "discussion-questions"], "chapters": 15},
    {"slug": "organizational-behavior", "subject": "business", "page_suffixes": ["chapter-review-questions", "critical-thinking-case"], "chapters": 19},
    {"slug": "principles-management", "subject": "business", "page_suffixes": ["chapter-review-questions", "critical-thinking-case"], "chapters": 18},
    {"slug": "principles-marketing", "subject": "business", "page_suffixes": ["critical-thinking-exercises"], "chapters": 19},
    {"slug": "principles-finance", "subject": "business", "page_suffixes": ["review-questions"], "chapters": 19},
    {"slug": "principles-financial-accounting", "subject": "business", "page_suffixes": ["questions", "thought-provokers"], "chapters": 16},
    {"slug": "principles-managerial-accounting", "subject": "business", "page_suffixes": ["questions", "thought-provokers"], "chapters": 13},
    {"slug": "introductory-business-statistics-2e", "subject": "business", "page_suffixes": ["chapter-review", "practice"], "chapters": 13},
    # ── Nursing / Health ──
    {"slug": "nutrition", "subject": "nursing", "page_suffixes": ["review-questions"], "chapters": 20},
    {"slug": "pharmacology", "subject": "nursing", "page_suffixes": ["review-questions"], "chapters": 40},
]


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_pages": [], "questions": []}


def save_progress(progress: dict):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)


def extract_questions(html: str, book_slug: str, chapter: int,
                      page_suffix: str, subject: str) -> list[dict]:
    """Extract questions from server-side rendered HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Find exercises — two formats exist
    exercises = soup.find_all(attrs={"data-type": "exercise"})
    exercises += soup.find_all(attrs={"data-type": "injected-exercise"})

    questions = []
    for ex in exercises:
        problem = ex.find(class_="os-problem-container")
        if not problem:
            continue

        # Extract stem (everything before the answer list)
        stem_parts = []
        for child in problem.children:
            if hasattr(child, "get"):
                dt = child.get("data-type", "")
                if dt in ("question-answers",):
                    break
            if hasattr(child, "name") and child.name in ("ol", "ul"):
                # Check if this is answer choices (type="a")
                if hasattr(child, "get") and child.get("type") == "a":
                    break
                # Regular list might be part of the stem
            text = child.get_text(strip=True) if hasattr(child, "get_text") else str(child).strip()
            if text:
                stem_parts.append(text)

        stem = " ".join(stem_parts).strip()
        if not stem:
            continue

        # Extract MCQ choices
        choices = []
        for li in problem.select('ol[type="a"] > li, [data-type="question-answers"] li, [data-type="question-answer"]'):
            choice_text = li.get_text(strip=True)
            if choice_text:
                choices.append(choice_text)

        # Extract solution if present
        solution = ex.find(class_="os-solution-container")
        answer = solution.get_text(strip=True) if solution else None

        questions.append({
            "question_text": stem,
            "question_type": "mcq" if choices else "open_ended",
            "choices": choices if choices else None,
            "correct_answer": answer,
            "subject": subject,
            "source_book": book_slug,
            "chapter": chapter,
            "source_page": f"{chapter}-{page_suffix}",
        })

    return questions


def scrape_page(book_slug: str, chapter: int, page_suffix: str,
                subject: str) -> list[dict]:
    """Fetch and extract questions from a single page."""
    url = f"https://openstax.org/books/{book_slug}/pages/{chapter}-{page_suffix}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
    except requests.RequestException:
        return []

    return extract_questions(resp.text, book_slug, chapter, page_suffix, subject)


def main():
    print("=" * 60, flush=True)
    print("CogBench — Scrape OpenStax Questions (HTTP/SSR)", flush=True)
    print("=" * 60, flush=True)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            existing = json.load(f)
        print(f"\nAlready exists: {OUTPUT_FILE} ({len(existing)} questions)")
        print("Delete it to re-scrape.")
        return 0

    progress = load_progress()
    completed = set(progress["completed_pages"])
    all_questions = progress["questions"]

    print(f"Resuming with {len(all_questions)} questions from {len(completed)} pages.\n",
          flush=True)

    for book in BOOKS:
        suffixes = book["page_suffixes"]
        book_total = 0

        # Skip fully completed books
        all_done = all(
            f"{book['slug']}/{ch}-{s}" in completed
            for ch in range(1, book["chapters"] + 1)
            for s in suffixes
        )
        if all_done:
            existing = sum(1 for q in all_questions if q.get("source_book") == book["slug"])
            print(f"[SKIP] {book['slug']}: already done ({existing} Qs)", flush=True)
            continue

        print(f"\n{'='*50}", flush=True)
        print(f"Book: {book['slug']} ({book['subject']}) — {len(suffixes)} page type(s)",
              flush=True)
        print(f"{'='*50}", flush=True)

        for ch in range(1, book["chapters"] + 1):
            ch_total = 0
            for suffix in suffixes:
                page_key = f"{book['slug']}/{ch}-{suffix}"
                if page_key in completed:
                    existing = sum(1 for q in all_questions
                                   if q.get("source_page") == f"{ch}-{suffix}"
                                   and q.get("source_book") == book["slug"])
                    ch_total += existing
                    continue

                questions = scrape_page(book["slug"], ch, suffix, book["subject"])

                if questions:
                    all_questions.extend(questions)
                    ch_total += len(questions)

                completed.add(page_key)
                progress["completed_pages"] = list(completed)
                progress["questions"] = all_questions
                save_progress(progress)

                # Polite delay
                time.sleep(0.3)

            book_total += ch_total
            if ch_total > 0:
                print(f"  Ch {ch:2d}: {ch_total} questions", flush=True)

        print(f"  Subtotal for {book['slug']}: {book_total}", flush=True)

    # Save final output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"TOTAL: {len(all_questions)} questions scraped", flush=True)
    print(f"{'='*60}", flush=True)

    subject_counts = {}
    type_counts = {}
    for q in all_questions:
        subject_counts[q["subject"]] = subject_counts.get(q["subject"], 0) + 1
        type_counts[q["question_type"]] = type_counts.get(q["question_type"], 0) + 1

    print("\nBy subject:", flush=True)
    for subj, count in sorted(subject_counts.items()):
        print(f"  {subj}: {count}", flush=True)

    print("\nBy type:", flush=True)
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}", flush=True)

    print(f"\nSaved: {OUTPUT_FILE}", flush=True)

    # Clean up progress
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
