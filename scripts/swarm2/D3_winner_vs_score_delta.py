"""D3 secondary: localise round_end winner errors with the engine score delta (2024 G: tome).

For round r, the score delta is the change in round_state (t_score, ct_score) between round_freeze_end of
r+1 and of r. Both fields are side-based, so the delta gives the side that won round r, provided the sides
did not swap between the two freeze rows. Swaps follow the rule in bakeoff.swap_expected with max_rounds
= 24 (ASSUMPTION). Rounds with swap_expected(r) are excluded. Round_end winner codes are side codes too,
so the two are compared directly.

Reported: per round number, the count of rounds where the valid delta side disagrees with round_end winner,
split by day group (01..07 vs 12..25), and the count of matches with at least one such disagreement.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D3_winner_vs_score_delta.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
MAX_ROUNDS = 24  # ASSUMPTION


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


RE = g_read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
win = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}

cnt = collections.Counter()
by_round = collections.Counter()
by_round_tot = collections.Counter()
dis_matches = collections.defaultdict(set)
dis_examples = []
for (m, r), (t1, c1) in live.items():
    if (m, r) not in win or (m, r + 1) not in live:
        cnt["no_next_or_no_winner"] += 1
        continue
    if swap_expected(r, MAX_ROUNDS):
        cnt["skipped_swap_round"] += 1
        continue
    t2, c2 = live[(m, r + 1)]
    dt, dc = t2 - t1, c2 - c1
    if dc == 1 and dt == 0:
        ds = CT
    elif dt == 1 and dc == 0:
        ds = T
    else:
        cnt["delta_unclear"] += 1
        continue
    grp = "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"
    cnt[f"valid_{grp}"] += 1
    by_round_tot[(grp, r)] += 1
    if ds != win[(m, r)]:
        cnt[f"disagree_{grp}"] += 1
        by_round[(grp, r)] += 1
        dis_matches[grp].add(m)
        if len(dis_examples) < 8:
            dis_examples.append((m, r, win[(m, r)], ds, (t1, c1), (t2, c2)))

print("counts:")
for k, v in sorted(cnt.items()):
    print(f"  {k:<28} {v}")
print("\ndisagreement by round (grp, round): disagree / valid")
for g in ["day01_07", "day12_25"]:
    items = sorted([(r, by_round[(g, r)], by_round_tot[(g, r)]) for (gg, r) in by_round_tot if gg == g])
    print(f"  {g}:", [(r, d, n) for r, d, n in items if d > 0])
print("\nmatches with >=1 disagreement:", {g: len(s) for g, s in dis_matches.items()})
print("examples (match, round, winner_round_end, score_delta_side, freeze_r, freeze_r+1):")
for e in dis_examples:
    print(" ", e)
