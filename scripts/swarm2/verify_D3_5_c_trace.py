"""D3-5 verifier, step C: per-round trace of sample 2024 day 12-25 matches with round-9 lead disagreement.

Columns per round: round_end winner (side code), round_end tick, freeze_end tick, live t/ct (side fields),
round_state raw fields, round_state round_end-event scores, bomb outcome, derived lead vs live lead.
ASSUMPTION: derived lead uses MAX_ROUNDS=24 swap rule as in D3_lead_derivation_2024.py (only matters after round 12).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_width_chars(260)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


RE = rd("round_end", ["match_key", "round", "winner_team_code", "tick", "player_count", "win_reason_code"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
fe = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
fe1 = fe.group_by(["match_key", "round"]).agg(pl.col("tick").first().alias("fe_tick"), pl.col("t_score").first().alias("t"),
                                             pl.col("ct_score").first().alias("ct"),
                                             pl.col("t_score_raw").first().alias("t_raw"),
                                             pl.col("ct_score_raw").first().alias("ct_raw"))
rend = RS.filter(pl.col("event_type") == "round_end").sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("tick").first().alias("rs_end_tick"), pl.col("t_score").first().alias("t_end"),
    pl.col("ct_score").first().alias("ct_end"))
bomb = (BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().sort().alias("bevs")))

import sys as _s
targets = _s.argv[1:] if len(_s.argv) > 1 else []
mat = (RE.filter(pl.col("player_count") == 20).unique(["match_key", "round"], keep="first")
       .join(fe1, on=["match_key", "round"], how="left")
       .join(rend, on=["match_key", "round"], how="left")
       .join(bomb, on=["match_key", "round"], how="left")
       .sort(["match_key", "round"]))
# sample: day >= 12, contiguous from round 1, and first live-vs-derived disagreement at round 9
ms = mat.group_by("match_key").agg(pl.col("round").min().alias("mn")).filter(pl.col("mn") == 1)["match_key"]
mat = mat.filter(pl.col("match_key").is_in(ms.to_list()))
mat = mat.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
mat = mat.filter(pl.col("day") >= 12)
print("contiguous-from-1 day>=12 matches (player_count 20 rounds):", mat["match_key"].n_unique())

# derived lead over player_count-20 rounds in order (side A starts T, swap after round 12 and OT rule for MR24)
def swap_expected(r, mr=24):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)

rows = mat.select(["match_key", "round", "winner_team_code", "t", "ct"]).iter_rows()
per = {}
for m, r, w, t, c in rows:
    per.setdefault(m, []).append((r, w, t, c))
first_dis = {}
for m, lst in per.items():
    side_a, a, b = T, 0, 0
    for r, w, t, c in lst:
        if t is None or c is None:
            continue
        derived = (b - a) if side_a == T else (a - b)
        live_lead = c - t
        if derived != live_lead and m not in first_dis:
            first_dis[m] = r
        if w == side_a:
            a += 1
        elif w == (CT if side_a == T else T):
            b += 1
        if swap_expected(r):
            side_a = CT if side_a == T else T
print("matches with first disagreement at round 9 (day>=12, contiguous):",
      sum(1 for v in first_dis.values() if v == 9), "| at round 10:", sum(1 for v in first_dis.values() if v == 10))
samp9 = [m for m, v in first_dis.items() if v == 9][:3]
samp10 = [m for m, v in first_dis.items() if v == 10][:1]
for m in samp9 + samp10:
    print("\n==== MATCH", m, "first disagreement round", first_dis[m])
    sub = mat.filter(pl.col("match_key") == m).select(
        ["round", "winner_team_code", "tick", "fe_tick", "t", "ct", "t_raw", "ct_raw", "rs_end_tick", "t_end", "ct_end",
         "win_reason_code", "bevs"]).sort("round")
    print(sub.head(16))
