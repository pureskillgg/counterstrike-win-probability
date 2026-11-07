"""A9 audit probe (read-only): round_end coverage, 2024-02 vs 2026.

For every header match: rounds implied by the header final score (t_starters_score_final +
ct_starters_score_final) minus round_end rows. Positive means round_end is missing rounds (if the header
final score is the real total). Also counts round numbers absent from 1..max(round) in round_end.

Usage (project root): python.exe -B -I -X utf8 scripts\\swarm2\\A9_probe_coverage.py
"""

import collections
import importlib.util
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fv = load_module("feature_validation", ROOT / "scripts" / "model" / "feature_validation.py")


def coverage(tome, suffix, label):
    old = (fv.TOME, fv.SUFFIX)
    fv.TOME, fv.SUFFIX = tome, suffix
    try:
        H = fv.read("header", ["key", "t_starters_score_final", "ct_starters_score_final"])
        RE = fv.read("round_end", ["match_key", "round"]).unique(["match_key", "round"], keep="first")
    finally:
        fv.TOME, fv.SUFFIX = old
    agg = RE.group_by("match_key").agg(pl.len().alias("n"), pl.col("round").max().alias("mx"),
                                       pl.col("round").n_unique().alias("nu"))
    d = {m: (n, mx, nu) for m, n, mx, nu in agg.iter_rows()}
    diffs = collections.Counter()
    gap_numbers = 0
    matches_with_gap = 0
    total_missing = 0
    for r in H.iter_rows(named=True):
        t, c = r["t_starters_score_final"], r["ct_starters_score_final"]
        if t is None or c is None or r["key"] not in d:
            diffs["no score or no round_end"] += 1
            continue
        n, mx, nu = d[r["key"]]
        miss = int(t) + int(c) - n
        diffs["equal" if miss == 0 else ("rows short by 1" if miss == 1 else
                                         ("rows short by 2+" if miss > 1 else "rows over"))] += 1
        if miss > 0:
            total_missing += miss
            matches_with_gap += 1
        if mx > nu - 0:
            gap_numbers += 1
    n_m = H.height
    print(f"{label}: header matches {n_m}; round_end rows {RE.height}; "
          f"header-sum minus rows: {dict(diffs)}; total rounds short (positive gaps) {total_missing}; "
          f"matches with missing round numbers in 1..max: {gap_numbers}", flush=True)


def main():
    coverage(ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08", "2026")
    coverage(ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25", "2024-02")


if __name__ == "__main__":
    main()
