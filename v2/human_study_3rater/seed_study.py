#!/usr/bin/env python3
"""Load items.json into the Classrooms MongoDB and mint one link per rater.

Usage:
  python seed_study.py                 # 3 raters: Rater A, Rater B, Rater C
  python seed_study.py --raters 4      # add more later (existing links kept)
  python seed_study.py --reset         # reload items (refuses if answers exist)

Links look like https://classrooms.work/cograte/<token> and are written to
links.txt. Raters see only the passage and question, never the answer key.
"""
import argparse
import json
import secrets
import string
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient

HERE = Path(__file__).resolve().parent
BASE_URL = "https://classrooms.work/cograte"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    db = MongoClient("localhost", 27017)["lectura_ai"]
    questions, sessions, answers = (db.cogbench_rating_questions, db.cogbench_rating_sessions,
                                    db.cogbench_rating_answers)

    if args.reset:
        n = answers.count_documents({})
        if n:
            raise SystemExit(f"Refusing to reset: {n} answers already saved")
        questions.delete_many({})

    if questions.count_documents({}) == 0:
        items = json.load(open(HERE / "items.json"))["items"]
        missing = [it["item_id"] for it in items if not it["question"]]
        if missing:
            raise SystemExit(f"Items without a question: {missing}; regenerate first")
        questions.insert_many([
            {"item_id": it["item_id"], "order": k, "passage_text": it["passage_text"],
             "question_text": it["question"]}
            for k, it in enumerate(items, 1)
        ])
        questions.create_index("item_id", unique=True)
        answers.create_index([("token", 1), ("item_id", 1)], unique=True)
        print(f"loaded {len(items)} items")

    have = sessions.count_documents({})
    for i in range(have, args.raters):
        sessions.insert_one({
            "token": secrets.token_urlsafe(16),
            "rater_label": f"Rater {string.ascii_uppercase[i]}",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    lines = [f"{s['rater_label']}: {BASE_URL}/{s['token']}"
             for s in sessions.find({}).sort("rater_label", 1)]
    (HERE / "links.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
