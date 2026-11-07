#!/usr/bin/env python3
"""Agreement and adherence analysis for the 3-rater CogBench human study.

Reads ratings/ratings_long.csv (written by export_ratings.py) and reports:
  - inter-rater agreement on Bloom level over items rated by all raters:
    Fleiss' kappa, pairwise Cohen's kappa (unweighted and quadratic),
    pairwise exact / adjacent (+-1) agreement, Krippendorff's alpha (ordinal)
  - majority-vote label (>= 2 raters agree; otherwise the median rating)
  - human-judged adherence to the target level, strict and adjacent, by mode
  - validity: share rated jointly answerable, faithful and coherent, by mode
Writes results.json and results.md next to this script.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEVELS = np.arange(1, 7)


def cohen(a, b, weights=None):
    a, b = np.asarray(a), np.asarray(b)
    k = len(LEVELS)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[x - 1, y - 1] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / (k - 1) ** 2 if weights == "quadratic" else (i != j).astype(float)
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom else float("nan")


def fleiss(matrix):
    """matrix: items x raters of labels 1..6."""
    n_items, n_raters = matrix.shape
    counts = np.stack([(matrix == lv).sum(1) for lv in LEVELS], axis=1)
    p_j = counts.sum(0) / (n_items * n_raters)
    p_i = ((counts ** 2).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def kripp_ordinal(matrix):
    """Krippendorff's alpha, ordinal metric, no missing values."""
    values = matrix.flatten()
    n_c = np.array([(values == lv).sum() for lv in LEVELS], dtype=float)
    cum = np.cumsum(n_c)

    def delta(c, k):
        lo, hi = min(c, k), max(c, k)
        # sum of n_g for g in [lo, hi] minus (n_lo + n_hi) / 2, squared
        return (cum[hi - 1] - cum[lo - 1] + n_c[lo - 1] / 2 - n_c[hi - 1] / 2) ** 2 if lo != hi else 0.0

    d = np.array([[delta(c, k) for k in LEVELS] for c in LEVELS])
    m = matrix.shape[1]
    d_o = sum(d[x - 1, y - 1] for row in matrix for x, y in itertools.permutations(row, 2)) / (
        len(values) * (m - 1))
    n = len(values)
    d_e = sum(n_c[c] * n_c[k] * d[c, k] for c in range(6) for k in range(6)) / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e else float("nan")


def majority(row):
    vals, cnt = np.unique(row, return_counts=True)
    return int(vals[cnt.argmax()]) if cnt.max() >= 2 else int(np.median(row))


def main():
    df = pd.read_csv(HERE / "ratings" / "ratings_long.csv")
    raters = sorted(df.rater.unique())
    wide = df.pivot(index="item_id", columns="rater", values="bloom_level")
    full = wide.dropna().astype(int)
    key = df.drop_duplicates("item_id").set_index("item_id")[["mode", "target_level"]]
    out = {"raters": raters, "answers_per_rater": df.groupby("rater").size().to_dict(),
           "items_rated_by_all": int(len(full))}

    if len(raters) >= 2 and len(full):
        pairs = {}
        for r1, r2 in itertools.combinations(raters, 2):
            a, b = full[r1].values, full[r2].values
            pairs[f"{r1} vs {r2}"] = {
                "exact": float((a == b).mean()), "adjacent": float((abs(a - b) <= 1).mean()),
                "cohen_kappa": cohen(a, b), "cohen_kappa_quadratic": cohen(a, b, "quadratic")}
        out["pairwise"] = pairs
        m = full[raters].values
        out["fleiss_kappa"] = fleiss(m)
        out["krippendorff_alpha_ordinal"] = kripp_ordinal(m)
        out["all_agree_exact"] = float((m.max(1) == m.min(1)).mean())
        out["majority_exists"] = float(np.mean([np.unique(r, return_counts=True)[1].max() >= 2
                                                for r in m]))

        maj = full[raters].apply(lambda r: majority(r.values), axis=1).rename("majority")
        j = key.join(maj, how="inner")
        j["strict"] = j.majority == j.target_level
        j["adjacent"] = (j.majority - j.target_level).abs() <= 1
        out["majority_adherence"] = {
            mode: {"n": int(len(g)), "strict": float(g.strict.mean()), "adjacent": float(g.adjacent.mean())}
            for mode, g in j.groupby("mode")}
        # Meta-review W1: how often is a level-correct generation also a valid question?
        valid_item = (df.assign(v=df.answerable & df.faithful & df.coherent)
                        .groupby("item_id").v.mean() >= 0.5)  # majority of raters judged it valid
        correct = j[j.strict].index
        out["valid_given_level_correct"] = {
            "n_items": int(len(correct)),
            "share_valid": float(valid_item.reindex(correct).mean()) if len(correct) else float("nan")}

    # Rating pace per rater (gaps between consecutive saves, in seconds).
    pace = {}
    for rater, g in df.groupby("rater"):
        t = pd.to_datetime(g.answered_at).sort_values()
        gaps = t.diff().dt.total_seconds().dropna()
        pace[rater] = {"minutes_total": float((t.iloc[-1] - t.iloc[0]).total_seconds() / 60),
                       "median_seconds_per_item": float(gaps.median()) if len(gaps) else float("nan"),
                       "gaps_under_5s": int((gaps < 5).sum())}
    out["pace"] = pace

    df["valid"] = df.answerable & df.faithful & df.coherent
    out["validity"] = {
        "overall": float(df.valid.mean()),
        "by_mode": df.groupby("mode").valid.mean().to_dict(),
        "answerable": float(df.answerable.mean()), "faithful": float(df.faithful.mean()),
        "coherent": float(df.coherent.mean())}

    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = [f"# 3-rater CogBench human study results", "",
             f"Raters: {', '.join(raters)}; answers per rater: {out['answers_per_rater']}; "
             f"items rated by all: {out['items_rated_by_all']}", ""]
    if "fleiss_kappa" in out:
        lines += [f"- Fleiss' kappa: {out['fleiss_kappa']:.3f}",
                  f"- Krippendorff's alpha (ordinal): {out['krippendorff_alpha_ordinal']:.3f}",
                  f"- All raters agree exactly: {out['all_agree_exact']:.1%}; "
                  f"majority (>=2) exists: {out['majority_exists']:.1%}", ""]
        for p, v in out["pairwise"].items():
            lines.append(f"- {p}: exact {v['exact']:.1%}, adjacent {v['adjacent']:.1%}, "
                         f"kappa {v['cohen_kappa']:.3f}, quadratic kappa {v['cohen_kappa_quadratic']:.3f}")
        lines.append("")
        for mode, v in out["majority_adherence"].items():
            lines.append(f"- Majority-vote adherence, {mode} (n={v['n']}): strict {v['strict']:.1%}, "
                         f"adjacent {v['adjacent']:.1%}")
        vg = out["valid_given_level_correct"]
        lines.append(f"- Valid among majority-level-correct items: {vg['share_valid']:.1%} (n={vg['n_items']})")
    lines.append("")
    for rater, p in out["pace"].items():
        lines.append(f"- Pace, {rater}: {p['minutes_total']:.1f} min total, median {p['median_seconds_per_item']:.0f} s "
                     f"per item, {p['gaps_under_5s']} gaps under 5 s")
    v = out["validity"]
    lines += ["", f"- Jointly answerable+faithful+coherent: {v['overall']:.1%} "
              f"({', '.join(f'{k} {x:.1%}' for k, x in v['by_mode'].items())})"]
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
