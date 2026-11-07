#!/usr/bin/env python3
"""Dump the study's MongoDB collections into this folder (git-tracked).

Writes ratings/answers.json, ratings/sessions.json, ratings/questions.json and
ratings/ratings_long.csv (one row per rater x item, joined with the answer
key). Run by sync.sh every 10 minutes; safe to run any time.
"""
import csv
import json
from pathlib import Path

from pymongo import MongoClient

HERE = Path(__file__).resolve().parent
OUT = HERE / "ratings"


def dump(coll):
    return [{k: v for k, v in d.items() if k != "_id"} for d in coll.find({})]


def main():
    db = MongoClient("localhost", 27017)["lectura_ai"]
    OUT.mkdir(exist_ok=True)
    data = {n: dump(db[f"cogbench_rating_{n}"]) for n in ("sessions", "questions", "answers")}
    # Only the current links count; answers from retired links stay in the database
    # (see seed_study.py --new-round) but are not exported as ratings.
    live = {s["token"] for s in data["sessions"]}
    data["answers"] = [a for a in data["answers"] if a["token"] in live]
    for n, rows in data.items():
        rows.sort(key=lambda d: (d.get("token", ""), d.get("item_id", ""), d.get("rater_label", "")))
        (OUT / f"{n}.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")

    label = {s["token"]: s["rater_label"] for s in data["sessions"]}
    key = {r["item_id"]: r for r in csv.DictReader(open(HERE / "answer_key.csv"))}
    with open(OUT / "ratings_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "item_id", "bloom_level", "answerable", "faithful", "coherent",
                    "mode", "target_level", "vocab_level", "subject", "answered_at", "updated_at"])
        for a in sorted(data["answers"], key=lambda a: (label.get(a["token"], ""), a["item_id"])):
            k = key.get(a["item_id"], {})
            w.writerow([label.get(a["token"], a["token"]), a["item_id"], a["bloom_level"],
                        int(a["answerable"]), int(a["faithful"]), int(a["coherent"]),
                        k.get("mode"), k.get("target_level"), k.get("vocab_level"), k.get("subject"),
                        a.get("answered_at"), a.get("updated_at")])
    print(f"exported {len(data['answers'])} answers from {len(data['sessions'])} raters")


if __name__ == "__main__":
    main()
