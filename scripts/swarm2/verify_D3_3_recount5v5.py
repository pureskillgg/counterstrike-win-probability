"""verify D3-3: recount the claim's 2,572 disagreements by 5v5 status (2024 G: tome, read-only).

round_end.player_count in the 2024 tome is not 10 for 5v5 rounds (first rows show 20), so check its distribution
and count 5 T / 5 CT from player_info as the authoritative 5v5 test.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def chan_files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def scan(ch):
    return pl.scan_parquet(chan_files(ch), missing_columns="insert", extra_columns="ignore")


def grp(m):
    return "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"


RE = (scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
      .sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True))
print("round_end.player_count distribution:", RE.group_by("player_count").len().sort("player_count").rows())
W = {(m, r): (w, pc) for m, r, w, pc in RE.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows()}

RS = scan("round_state").filter(pl.col("event_type") == "round_freeze_end").select(
    ["match_key", "round", "tick", "t_score", "ct_score"]).collect()
FE = RS.sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first())
F = {(m, r): (t, c) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}

PI = scan("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
nT, nC = collections.Counter(), collections.Counter()
for m, r, pid, tc in PI.iter_rows():
    if tc == T:
        nT[(m, r)] += 1
    elif tc == CT:
        nC[(m, r)] += 1


def swap_claim(r, mr=24):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


D = {}
for (m, r), (t1, c1) in F.items():
    if (m, r + 1) not in F or swap_claim(r):
        continue
    t2, c2 = F[(m, r + 1)]
    dt, dc = t2 - t1, c2 - c1
    if dc == 1 and dt == 0:
        D[(m, r)] = CT
    elif dt == 1 and dc == 0:
        D[(m, r)] = T

cnt = collections.Counter()
per_round = collections.Counter()
for (m, r), d in D.items():
    if (m, r) not in W or grp(m) != "day12_25":
        continue
    w, pc = W[(m, r)]
    five = nT[(m, r)] == 5 and nC[(m, r)] == 5
    cnt["valid"] += 1
    cnt["valid_5v5"] += int(five)
    if d != w:
        cnt["claim_disagree"] += 1
        cnt["claim_disagree_5v5"] += int(five)
        per_round[(r, "5v5" if five else "not5v5")] += 1
print("\nday12_25 counts:", dict(cnt))
print("claim disagreements by round and 5v5 status:")
for r in (9, 13, 25, 28):
    print(f"  r={r}: 5v5 {per_round[(r, '5v5')]}, not5v5 {per_round[(r, 'not5v5')]}")
print("player_info T/CT counts over claim-disagreement rounds:",
      collections.Counter((nT[(m, r)], nC[(m, r)]) for (m, r), d in D.items()
                          if (m, r) in W and grp(m) == "day12_25" and d != W[(m, r)][0]).most_common(8))
