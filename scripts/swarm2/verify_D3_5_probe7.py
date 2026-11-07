"""Verifier D3-5, probe 7: the 8-swap matches (physical side swap after completed round 8) in 2024.
Header score totals, round counts, death-feed coverage, bomb coverage, and round-9 comparisons. Read only.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


HD = pl.concat([pl.read_parquet(f, columns=["key", "t_starters_score_final", "ct_starters_score_final"])
                for f in sorted(glob.glob(str(TOME / "header" / "dataframe_*")))])
HD = HD.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_total"))
PI = g_read("player_info", ["match_key", "round", "player_id", "team_code"])
RE = g_read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
DE = g_read("player_death", ["match_key", "round", "player_id", "player_team_code"])
BS = g_read("bomb_state", ["match_key", "round", "event_type"])

side = collections.defaultdict(lambda: {T: set(), CT: set()})
for m, r, pid, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    if tc in (T, CT):
        side[(m, int(r))][tc].add(pid)
phys = collections.defaultdict(list)
rounds_by_m = collections.defaultdict(list)
for (m, r) in side:
    rounds_by_m[m].append(r)
for m, rs in rounds_by_m.items():
    for r in sorted(rs):
        if (m, r + 1) in side:
            s0, s1 = side[(m, r)], side[(m, r + 1)]
            if s0[T] and s1[T] and s1[T] == s0[CT] and s1[CT] == s0[T]:
                phys[m].append(r)
cls8 = [m for m, v in phys.items() if 8 in v]
cls12 = [m for m, v in phys.items() if 12 in v and 8 not in v]
none = [m for m in rounds_by_m if m not in phys]
print("8-swap matches:", len(cls8), "| 12-swap (no 8):", len(cls12), "| no physical swap:", len(none))
grp = lambda m: "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"
print("8-swap by group:", collections.Counter(grp(m) for m in cls8))
print("12-swap by group:", collections.Counter(grp(m) for m in cls12))

hd = HD.to_dicts()
hmap = {x["key"]: x["hdr_total"] for x in hd}
print("\nheader total (T+CT final) distribution, 8-swap matches:", collections.Counter(hmap[m] for m in cls8).most_common(12))
print("header total distribution, 12-swap matches:", collections.Counter(hmap[m] for m in cls12).most_common(12))
rcount = RE.group_by("match_key").agg(pl.len().alias("n"), pl.col("round").max().alias("mx")).to_dicts()
rc = {x["match_key"]: (x["n"], x["mx"]) for x in rcount}
print("round_end rows, 8-swap matches (n, max round):", collections.Counter(rc.get(m, (0, 0)) for m in cls8).most_common(10))
print("round_end rows, 12-swap matches (n, max round):", collections.Counter(rc.get(m, (0, 0)) for m in cls12).most_common(10))
dd = collections.Counter(m for m in DE.select("match_key").to_series().to_list())
print("\ndeath rows per match, 8-swap: median", sorted(dd.get(m, 0) for m in cls8)[len(cls8) // 2] if cls8 else None,
      "| matches with zero death rows:", sum(1 for m in cls8 if dd.get(m, 0) == 0))
print("death rows per match, 12-swap: median", sorted(dd.get(m, 0) for m in cls12)[len(cls12) // 2] if cls12 else None,
      "| matches with zero death rows:", sum(1 for m in cls12 if dd.get(m, 0) == 0))
bb = collections.Counter(m for m in BS.select("match_key").to_series().to_list())
print("bomb rows per match, 8-swap: zero-row matches:", sum(1 for m in cls8 if bb.get(m, 0) == 0),
      "| 12-swap zero-row matches:", sum(1 for m in cls12 if bb.get(m, 0) == 0))
pi_sizes = collections.Counter((len(side[(m, r)][T]), len(side[(m, r)][CT])) for m in cls8 for r in rounds_by_m[m])
print("(nT, nCT) per round in 8-swap matches:", pi_sizes.most_common(6))
