"""A9 audit probes (read-only):
  Q1  Round-13 defect flags (bomb outcome vs round_end winner_team_code) on the 2026 kept set,
      using the same rule as A9_cross_period_r13.py. The bake-off and every 2026-based number would be
      affected if this is non-zero.
  Q2  max_rounds check: distribution of the maximum round_end round for matches whose max_rounds is
      16 (2026: header value; 2024: inferred from the halftime event) and 24 (same sources).

Usage (project root): python.exe -B -I -X utf8 scripts\\swarm2\\A9_probe_r13_2026_maxround.py
"""

import collections
import importlib.util
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
HALF_EVENT = "round_announce_last_round_half"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cp = load_module("cross_period", ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py")
fv = cp.fv


def flags_for(tome, suffix):
    old = (fv.TOME, fv.SUFFIX)
    fv.TOME, fv.SUFFIX = tome, suffix
    try:
        RE = fv.read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
        BS = fv.read("bomb_state", ["match_key", "round", "event_type"])
        RS = fv.read("round_state", ["match_key", "round", "event_type"])
        H = fv.read("header", ["key", "max_rounds"]) if suffix.startswith(".2026") else \
            fv.read("header", ["key", "match_date"])
    finally:
        fv.TOME, fv.SUFFIX = old
    return RE, BS, RS, H


def main():
    # Q1: 2026 round-13 rule on every round with a bomb outcome (no kept-set restriction)
    RE, BS, RS, H = flags_for(ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08")
    RE = RE.unique(["match_key", "round"], keep="first")
    outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    j = (RE.join(outcome, on=["match_key", "round"], how="inner")
         .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
                       .otherwise(pl.lit(2)).alias("bomb_side"))
         .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
    print(f"Q1 2026 rounds with a bomb outcome: {j.height}; mismatches: {int(j['mismatch'].sum())}; "
          f"round 13 rows: {j.filter(pl.col('round') == 13).height}, mismatches "
          f"{int(j.filter(pl.col('round') == 13)['mismatch'].sum())}")

    # Q2: max round_end round per match vs inferred or header max_rounds
    mr26 = dict(H.iter_rows())
    mx26 = dict(RE.group_by("match_key").agg(pl.col("round").max().alias("mx")).iter_rows())
    by = collections.defaultdict(list)
    for m, v in mr26.items():
        if m in mx26:
            by[v].append(mx26[m])
    for v in sorted(by):
        xs = by[v]
        print(f"Q2 2026 header max_rounds {v}: matches {len(xs)}; max round_end round: min {min(xs)}, "
              f"median {sorted(xs)[len(xs) // 2]}, max {max(xs)}; share with max round > {v}: "
              f"{sum(1 for x in xs if x > v) / len(xs):.3f}")

    RE4, BS4, RS4, H4 = flags_for(cp.TOME_2024, cp.SUFFIX_2024)
    half = {m: int(r) for m, r in (RS4.filter(pl.col("event_type") == HALF_EVENT)
                                    .group_by("match_key").agg(pl.col("round").min().alias("r"))
                                    .iter_rows())}
    mx24 = dict(RE4.group_by("match_key").agg(pl.col("round").max().alias("mx")).iter_rows())
    by24 = collections.defaultdict(list)
    for m, r in half.items():
        if m in mx24:
            by24[2 * r + 2].append(mx24[m])
    for v in sorted(by24):
        xs = by24[v]
        print(f"Q2 2024 inferred max_rounds {v} (halftime event): matches {len(xs)}; max round_end round: "
              f"min {min(xs)}, median {sorted(xs)[len(xs) // 2]}, max {max(xs)}; share with max round > {v}: "
              f"{sum(1 for x in xs if x > v) / len(xs):.3f}")


if __name__ == "__main__":
    main()
