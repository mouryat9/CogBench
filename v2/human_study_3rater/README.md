# CogBench 3-rater human study (November 2026)

> **Status:** complete (2026-11-07). Three raters rated all 48 items; results are in `results.md`
> (Fleiss' kappa 0.74; majority-vote adherence 91.7% standard vs 20.8% adversarial). Ratings are in
> `ratings/ratings_long.csv` (pseudonymous rater labels). A first round of ratings was discarded because
> its save timestamps were incompatible with reading the items; the study was rerun with fresh links.

A follow-up to the two-annotator study reported in the NeurIPS 2026 paper (submission #3477). It uses a
new item set rated by three raters; its results are not part of the camera-ready paper.

## Protocol

- **Items:** 48 generations = 6 Bloom levels x 2 modes (standard, adversarial) x
  4 passages. Each (level, mode) cell uses 4 different subjects; each of the 8
  subjects appears 6 times. Passages are drawn from the 120 CogBench passages
  with seed 20261107.
- **Generator:** `qwen2.5:14b` (one of the paper's 13 models) via Ollama, with the
  paper's own prompts, adversarial pairings ((2,1), (3,2), (4,1), (5,2), (6,3),
  (1,4)) and temperature 0.7 (`cogbenchv2.generation`).
- **Blinding:** items are shuffled and shown as C01..C48 with only the passage
  and the question. Raters never see the model, the mode, the target level or the
  verb whitelist. The key is in `answer_key.csv`.
- **Ratings per item:** perceived Bloom level (1-6) and three yes/no checks:
  answerable, faithful to the passage, coherent. The checks default to yes; one
  click on a level saves the item. Same definitions as the paper's Table 1.
- **Raters:** 3, working independently through personal links
  (`https://classrooms.work/cograte/<token>`, see `links.txt`). Every click is
  saved immediately; raters can resume at any time.
- **Analysis (`analyze.py`):** Fleiss' kappa, pairwise Cohen's kappa (unweighted
  and quadratic), pairwise exact and adjacent agreement, Krippendorff's alpha
  (ordinal), majority-vote label (>= 2 of 3 agree, else the median), human-judged
  strict / adjacent adherence by mode, and validity rates.

## Files

| File | What |
|---|---|
| `generate_items.py` | builds the item set: `items.json` (with prompts and raw outputs) and `answer_key.csv` |
| `seed_study.py` | loads the items into the Classrooms MongoDB (`lectura_ai.cogbench_rating_*`) and writes `links.txt` |
| `export_ratings.py` | dumps the collections into `ratings/` (JSON + `ratings_long.csv`) |
| `analyze.py` | writes `results.json` and `results.md` |
| `sync.sh` | export + analyze + git commit + push; runs from cron every 10 minutes |

The rating web page lives in the Classrooms repo
(`frontend/src/pages/shared/CogBenchRatingPage.tsx`,
`backend/api/routes/cogbench_rating.py`).
