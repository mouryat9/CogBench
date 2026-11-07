# CogBench 3-rater human study (November 2026)

> **Status:** rating in progress. `items.json`, `answer_key.csv` and the ratings are withheld
> until all three raters finish, so no rater can see the target levels; they will be added here
> afterwards. `sync.sh` and `seed_study.py` are specific to the server that hosts the rating page.

Added for the NeurIPS 2026 camera-ready (submission #3477). The meta-review made
acceptance conditional on, among other things, an expanded human evaluation with a
third annotator so that inter-annotator agreement could be computed over three
raters with majority adjudication. The items from the rebuttal-period study were
lost in the October 2026 server wipe, so this is a **new item set** rated by all
three raters, not a re-rating of the original 40 shared items. Report it that way.

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

## Still to record (needed for the paper and checklist)

- Who the three raters are (expertise), how they were recruited, and how they
  were compensated.
- IRB status for this study.
