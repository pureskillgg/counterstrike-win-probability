"""D3 secondary: which field disagrees on the 2024 round-13 mismatches?

Uses the G: 2024 tome. The four channels used here (round_end, round_state, bomb_state) are verified
row-identical to the E: tail copy for the 738 round-13 matches (see D3_compare_tome_vs_raw.py all mode).

For every (match, round) with a bomb outcome in round_end (player_count == 20, as in e2_side_check_r13.py):
  - winner_team_code (round_end)
  - bomb side: 3 if bomb_defused, 2 if bomb_exploded (planter side 2 = T is printed separately)
  - reason side from win_reason_code: 1 Target_Bombed, 9 Terrorists_Win, 13 Hostages_Not_Rescued,
    17 Terrorists_Surrender -> T (2); 7 Bomb_Defused, 8 CTs_Win, 11 All_Hostages_Rescued,
    12 Target_Saved, 18 CTs_Surrender -> CT (3). Mapping by reason name (ASSUMPTION: names as in
    feature_validation.WIN_REASON; 12 Target_Saved taken as CT win).
  - score side: round_freeze_end t_score/ct_score of round r+1 minus round r (ASSUMPTION: the score
    change across the round is the round winner; rounds with no next freeze_end are unclear).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D3_r13_label_source.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
REASON_SIDE = {1: T, 9: T, 13: T, 17: T, 7: CT, 8: CT, 11: CT, 12: CT, 18: CT}


def g_read(ch, cols=None):
    folder = TOME / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
print("round_end rows", RE.height, "bomb_state rows", BS.height, "round_state rows", RS.height)

# bomb outcome and planter side per (match, round)
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
plant = (BS.filter(pl.col("event_type") == "bomb_planted").sort("tick")
         .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter")))

# score at freeze end (first row per match, round)
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
fe_next = fe.with_columns((pl.col("round") - 1).alias("round_prev")).select(
    ["match_key", "round_prev", pl.col("t_score").alias("t_after"), pl.col("ct_score").alias("ct_after")])
fe_cur = fe.select(["match_key", "round", pl.col("t_score").alias("t_before"), pl.col("ct_score").alias("ct_before")])
score = fe_cur.join(fe_next, left_on=["match_key", "round"], right_on=["match_key", "round_prev"], how="left")
score = score.with_columns(
    (pl.col("t_after") - pl.col("t_before")).alias("dt"),
    (pl.col("ct_after") - pl.col("ct_before")).alias("dct"))
score = score.with_columns(
    pl.when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(CT))
    .when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(T))
    .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side"))

base = (RE.filter(pl.col("player_count") == 20)
        .join(outc, on=["match_key", "round"], how="inner")
        .join(plant, on=["match_key", "round"], how="left")
        .join(score.select(["match_key", "round", "score_side", "dt", "dct"]), on=["match_key", "round"], how="left")
        .with_columns(
            pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"),
            pl.col("win_reason_code").replace_strict(REASON_SIDE, default=None, return_dtype=pl.Int64).alias("reason_side"),
            pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"),
            pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
            .otherwise(pl.lit("other")).alias("rbin"))
        .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm")))

print(f"\nrounds with bomb outcome (player_count==20): {base.height}; mismatches (bomb side != winner): "
      f"{int(base['mm'].sum())}")
print("planter side counts on mismatch rounds:", base.filter(pl.col("mm"))["planter"].value_counts().rows())
print("planter side counts on match rounds:", base.filter(~pl.col("mm"))["planter"].value_counts().rows())


def agree(df, a, b):
    d = df.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
    return d.height, int((d[a] == d[b]).sum())


print("\n== agreement of the three sources, by round bin ==")
for rb in ["r13", "r25", "other"]:
    sub = base.filter(pl.col("rbin") == rb)
    for mmflag in [True, False]:
        s = sub.filter(pl.col("mm") == mmflag)
        if s.height == 0:
            continue
        n1, a1 = agree(s, "reason_side", "winner_team_code")
        n2, a2 = agree(s, "reason_side", "bomb_side")
        n3, a3 = agree(s, "score_side", "winner_team_code")
        n4, a4 = agree(s, "score_side", "bomb_side")
        print(f"{rb:<6} mismatch={str(mmflag):<5} n={s.height:>4} | "
              f"reason=winner {a1}/{n1} | reason=bomb {a2}/{n2} | score=winner {a3}/{n3} | score=bomb {a4}/{n4}")

print("\nwin_reason_code distribution, round-13 mismatch rows:")
print(base.filter((pl.col("rbin") == "r13") & pl.col("mm")).group_by("win_reason_code").len().sort("len", descending=True))
print("\nround-13 mismatch rows: (bomb_side, winner_team_code, reason_side) counts:")
print(base.filter((pl.col("rbin") == "r13") & pl.col("mm"))
      .group_by(["bomb_side", "winner_team_code", "reason_side"]).len().sort("len", descending=True))
print("\nround-13 mismatch rows: score_side vs winner_team_code vs bomb_side:")
print(base.filter((pl.col("rbin") == "r13") & pl.col("mm"))
      .group_by(["score_side", "winner_team_code", "bomb_side"], maintain_order=True).len())
print("\nround-13 agreeing rows (no mismatch), same grouping:")
print(base.filter((pl.col("rbin") == "r13") & ~pl.col("mm"))
      .group_by(["score_side", "winner_team_code", "bomb_side"], maintain_order=True).len())

print("\nscore-side coverage (rounds with bomb outcome and a score change):")
print(base.group_by(["rbin", "mm"]).agg(pl.len().alias("n"), pl.col("score_side").is_not_null().sum().alias("has_score")))

# does the flip hit round 13 only, or the whole second half? mismatch rate by round number for affected days
aff = base.filter(pl.col("day") >= 12)
print("\nmismatch rate by round number, day 12..25 (round: n, mm):")
rate = aff.group_by("round").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("round")
print(rate.rows())
