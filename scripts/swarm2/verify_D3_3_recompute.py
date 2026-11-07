"""verify D3-3: independent recompute of the claimed side-delta vs round_end winner counts (2024 G: tome).

Same definition as the claim, re-implemented from scratch:
  - live start = first round_freeze_end row per (match, round) by tick
  - winner = round_end winner_team_code, first row per (match, round)
  - delta side from freeze(r+1) - freeze(r) on (t_score, ct_score), only if exactly one side +1 and other side 0
  - skip swap_expected(r, 24) (ASSUMPTION, claim's rule), skip missing next round
Also reports the same counts split by 16-vs-24 round structure from player_info (players per round),
and by whether the round number is in the claimed disagreement set.
Read-only.
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
MAX_ROUNDS = 24  # claim's ASSUMPTION


def chan_files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def scan(ch):
    return pl.scan_parquet(chan_files(ch), missing_columns="insert", extra_columns="ignore")


def swap_claim(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def day_of(m):
    return int(m.split("/")[3])


RE_raw = scan("round_end").select(["match_key", "round", "tick", "winner_team_code"]).collect()
RE = (RE_raw.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True))
print(f"round_end rows {RE_raw.height}, distinct (match,round) {RE.height}")
win = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}

RS_raw = scan("round_state").filter(pl.col("event_type") == "round_freeze_end").select(
    ["match_key", "round", "tick", "t_score", "ct_score"]).collect()
dups = RS_raw.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
print(f"freeze_end rows {RS_raw.height}, (match,round) with >1 freeze row: {dups}")
fe = RS_raw.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("t_score").first(), pl.col("ct_score").first())
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}

# players per round (structure check): 10-player 5v5 rounds vs other
PI = scan("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
pc = PI.group_by(["match_key", "round"]).agg(
    pl.col("player_id").n_unique().alias("n_players"),
    (pl.col("team_code") == T).sum().alias("n_t"),
    (pl.col("team_code") == CT).sum().alias("n_ct"))
ten = {(m, r): (n, a, b) for m, r, n, a, b in pc.select(["match_key", "round", "n_players", "n_t", "n_ct"]).iter_rows()}

cnt = collections.Counter()
by_round = collections.Counter()
by_round_tot = collections.Counter()
by_round_mr16_tot = collections.Counter()
for (m, r), (t1, c1) in live.items():
    if (m, r) not in win or (m, r + 1) not in live:
        cnt["no_next_or_no_winner"] += 1
        continue
    if swap_claim(r, MAX_ROUNDS):
        cnt["skipped_swap_round_claim"] += 1
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
    grp = "day12_25" if day_of(m) >= 12 else "day01_07"
    cnt[f"valid_{grp}"] += 1
    by_round_tot[(grp, r)] += 1
    npl = ten.get((m, r), (None, None, None))[0]
    if npl != 10:
        cnt[f"valid_{grp}_non10players"] += 1
    if ds != win[(m, r)]:
        cnt[f"disagree_{grp}"] += 1
        by_round[(grp, r)] += 1
        if npl != 10:
            cnt[f"disagree_{grp}_non10players"] += 1

print("\ncounts (claim definition, re-implemented):")
for k, v in sorted(cnt.items()):
    print(f"  {k:<36} {v}")
print("\ndisagreement by round (grp, round): disagree / valid")
for g in ["day01_07", "day12_25"]:
    items = sorted([(r, by_round[(g, r)], by_round_tot[(g, r)]) for (gg, r) in by_round_tot if gg == g])
    print(f"  {g}: nonzero ->", [(r, d, n) for r, d, n in items if d > 0])
    print(f"  {g}: all rounds 1..30 ->", [(r, d, n) for r, d, n in items if r <= 30])

# player-count structure of the whole 2024 set
print("\nplayers-per-round distribution across all (match,round) in player_info:")
print(pc.group_by("n_players").len().sort("n_players").rows())
print("\nmatches by max round number in round_end (day12_25 vs day01_07):")
mx = RE.group_by("match_key").agg(pl.col("round").max().alias("max_round"), pl.len().alias("n_rounds"))
mx = mx.with_columns(pl.col("match_key").str.split("/").list.get(3).cast(pl.Int64).alias("day"))
print("max_round dist (day01_07):", mx.filter(pl.col("day") < 12).group_by("max_round").len().sort("max_round").rows())
print("max_round dist (day12_25):", mx.filter(pl.col("day") >= 12).group_by("max_round").len().sort("max_round").rows())
print("matches:", mx.height, " day01_07:", mx.filter(pl.col("day") < 12).height,
      " day12_25:", mx.filter(pl.col("day") >= 12).height)
