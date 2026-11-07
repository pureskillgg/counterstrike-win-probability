"""verify D5-9, step 8: 2v2 format facts compared across the 2024 and 2026 tomes.

For 4-player matches in each tome (2024 regime A, 2024 regime B, 2026 A):
  * engine phase == 'halftime' round labels (where the engine itself is at halftime)
  * maximum team freeze-end score, and how many matches have a team at >= 8 at a freeze-end that is
    followed by another round (first-to-8 predicts 0; first-to-9 is consistent)
  * round_end player_count values (2024 regime A 2v2 shows 20; check 2026 2v2)
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_08_format_compare.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

TOMES = {
    "2024": (Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"), ".2023-12-28,2024-02-25"),
    "2026": (Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"), ".2026-09-18,2026-10-08"),
}


def scan(base, suf, ch, cols):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in fs], how="vertical_relaxed").collect()


for tome, (base, suf) in TOMES.items():
    print("=" * 100)
    print("TOME", tome)
    RS = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
    PI = scan(base, suf, "player_info", ["match_key", "player_id"])
    RE = scan(base, suf, "round_end", ["match_key", "round", "winner_team_code", "player_count"])
    npl = PI.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n")).filter(pl.col("n") == 4)
    two = set(npl["match_key"].to_list())
    print("  4-player matches:", len(two))
    print("  round_end player_count for 4-player matches:", dict(Counter(
        RE.filter(pl.col("match_key").is_in(list(two)))["player_count"].to_list()).most_common(6)))
    ht = (RS.filter((pl.col("phase") == "halftime") & pl.col("match_key").is_in(list(two)))
          .group_by("match_key").agg(pl.col("round").unique().sort().alias("ht_labels")))
    print("  engine halftime-phase labels (4p matches, top):",
          Counter(",".join(str(x) for x in l) for l in ht["ht_labels"].to_list()).most_common(5))
    fe = (RS.filter((pl.col("event_type") == "round_freeze_end") & pl.col("match_key").is_in(list(two)))
          .sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
              pl.col("t_score").first().alias("t"), pl.col("ct_score").first().alias("c")))
    per = defaultdict(dict)
    for r in fe.iter_rows(named=True):
        per[r["match_key"]][r["round"]] = (r["t"], r["c"])
    reached_before_last = 0
    max_hist = Counter()
    n = 0
    for mk, d in per.items():
        labs = sorted(d)
        if len(labs) < 2:
            continue
        n += 1
        mx = max(max(d[l]) for l in labs)
        max_hist[mx] += 1
        if any(max(d[l]) >= 8 for l in labs[:-1]):
            reached_before_last += 1
    print("  matches with >=2 freeze-ends:", n)
    print("  max team freeze-end score distribution:", dict(sorted(max_hist.items())))
    print("  matches where a team had >=8 at a freeze-end followed by another round:", reached_before_last)
    print("  (first-to-8 predicts 0; a 9-win threshold allows it)")
    print("  matches where a team had >=9 at any freeze-end:", sum(1 for mk, d in per.items()
                                                              if max(max(v) for v in d.values()) >= 9))

print("\nDONE format compare")
