"""verify D9-8: independent recompute of the 2024 round-13 bomb-outcome mismatch and the
win_reason_message claim. Read-only on the tome; column projection only; no output written
under E:.

Sources (2024-02 tome, one lazy scan per table):
  header      : build_num, match_date, final team scores
  round_end   : winner_team_code, win_reason_code, win_reason_message, player_count, tick
  bomb_state  : bomb_defused / bomb_exploded rows (same outcome rule as e2_side_check_r13.py)
  round_state : t_score / ct_score on round_freeze_end (pre-round) and round_end (post-round)

ASSUMPTIONS (marked, not verified beyond the checks printed here):
  A1  team codes: 2 = T, 3 = CT (e2 docstring; the 2026 tome has 0 mismatches under this rule)
  A2  bomb side: bomb_defused -> 3 (CT), bomb_exploded -> 2 (T); defused wins if both present
  A3  score winner: side whose score rose by exactly 1 between round_freeze_end and round_end
      rows of the same round (both rows precede the halftime swap, so the frame is constant)
  A4  match day from match_key path csds/2024/02/DD/ (e2 rule)
  A5  generic message = #SFUI_Notice_Terrorists_Win or #SFUI_Notice_CTs_Win
  A6  reconciliation population = player_count == 20 and a bomb outcome row (e2 filter)
  A7  duplicate (match, round) rows are kept as in e2 for reconciliation; counted separately
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
GENERIC = ["#SFUI_Notice_Terrorists_Win", "#SFUI_Notice_CTs_Win"]
SPECIFIC = ["#SFUI_Notice_Bomb_Defused", "#SFUI_Notice_Target_Bombed"]


def files_of(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


def lz(name, cols):
    lf = pl.scan_parquet(files_of(name))
    have = lf.collect_schema().names()
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"{name} missing columns {miss}")
    return lf.select(cols)


def sec(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ----------------------------------------------------------------- header
H = lz("header", ["key", "build_num", "match_date", "map_name",
                  "t_starters_score_final", "ct_starters_score_final"]).collect()
H = H.rename({"key": "match_key"}).with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
    pl.col("match_date").str.slice(8, 2).alias("date_day"),
)
sec("S0 header population")
print("header rows", H.height, "| unique match_key", H["match_key"].n_unique(),
      "| day parse nulls", H.filter(pl.col("day").is_null()).height)
print("build_num x day:")
print(H.group_by(["build_num", "day"]).len().sort(["day", "build_num"]).rows())
print("day (path) vs match_date day agree:", int((H["day"] == H["date_day"]).sum()), "of", H.height)

# ----------------------------------------------------------------- round_end
RE = lz("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                      "win_reason_message", "player_count"]).collect()
dup = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
sec("S0b round_end population")
print("round_end rows", RE.height, "| (match, round) groups with >1 row:", dup.height,
      "| extra rows:", int((dup["len"] - 1).sum()) if dup.height else 0)
print("winner_team_code counts (all rows):", RE.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("win_reason_message counts (all rows):")
print(RE.group_by("win_reason_message").len().sort("len", descending=True))

# ----------------------------------------------------------------- bomb outcome
BO = lz("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect()
OUT = BO.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_defused").any().alias("defused"),
    (pl.col("event_type") == "bomb_exploded").any().alias("exploded"),
    pl.len().alias("n_out"),
    pl.col("tick").min().alias("bomb_tick"),
)
print("\nbomb outcome rows:", BO.height, "| (match, round) with outcome:", OUT.height,
      "| rounds with both defused and exploded:", OUT.filter(pl.col("defused") & pl.col("exploded")).height,
      "| rounds with >1 outcome row:", OUT.filter(pl.col("n_out") > 1).height)

# ----------------------------------------------------------------- round_state scores
RS = lz("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type").is_in(["round_freeze_end", "round_end"])).collect()
fe = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("tick").first().alias("fe_tick"),
    pl.col("t_score").first().alias("t_pre"),
    pl.col("ct_score").first().alias("ct_pre"),
    pl.len().alias("n_fe"),
)
re_ = RS.filter(pl.col("event_type") == "round_end").sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("tick").last().alias("rs_end_tick"),
    pl.col("t_score").last().alias("t_post"),
    pl.col("ct_score").last().alias("ct_post"),
    pl.len().alias("n_re"),
)
SW = fe.join(re_, on=["match_key", "round"], how="full", coalesce=True).with_columns(
    (pl.col("t_post") - pl.col("t_pre")).alias("dT"),
    (pl.col("ct_post") - pl.col("ct_pre")).alias("dCT"),
)
SW = SW.with_columns(
    pl.when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
      .when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_winner"),
)
sec("S0c round_state coverage and score-delta classes (A3)")
print("round_state freeze_end groups:", fe.height, "| round_end groups:", re_.height)
print("groups with n_fe != 1:", fe.filter(pl.col("n_fe") != 1).height,
      "| groups with n_re != 1:", re_.filter(pl.col("n_re") != 1).height)
print("(dT, dCT) classes over all round_state rounds with both rows:")
print(SW.filter(pl.col("dT").is_not_null() & pl.col("dCT").is_not_null())
      .group_by(["dT", "dCT"]).len().sort(["dT", "dCT"]))

# ----------------------------------------------------------------- round table
R = (RE.join(OUT, on=["match_key", "round"], how="left")
       .join(SW.select(["match_key", "round", "fe_tick", "rs_end_tick", "t_pre", "ct_pre", "t_post",
                        "ct_post", "dT", "dCT", "score_winner"]),
             on=["match_key", "round"], how="left")
       .join(H.select(["match_key", "build_num", "day", "map_name", "t_starters_score_final",
                       "ct_starters_score_final"]), on="match_key", how="left"))
R = R.with_columns(
    pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False),
)
R = R.with_columns(
    (pl.col("defused") | pl.col("exploded")).alias("has_out"),
    pl.when(pl.col("defused")).then(pl.lit(3)).when(pl.col("exploded")).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
      .otherwise(pl.lit("other")).alias("rbin"),
    pl.when(pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("01-07"))
      .otherwise(pl.lit("12-25")).alias("dayg"),
    pl.col("win_reason_message").is_in(GENERIC).alias("generic"),
    pl.col("win_reason_message").is_in(SPECIFIC).alias("specific"),
)
R = R.with_columns(
    pl.when(pl.col("has_out")).then(pl.col("bomb_side") != pl.col("winner_team_code"))
      .otherwise(None).alias("m1_e2"),
    pl.when(pl.col("score_winner").is_not_null()).then(pl.col("score_winner") != pl.col("winner_team_code"))
      .otherwise(None).alias("m2_score_vs_roundend"),
    pl.when(pl.col("has_out") & pl.col("score_winner").is_not_null())
      .then(pl.col("score_winner") != pl.col("bomb_side")).otherwise(None).alias("m3_score_vs_bomb"),
)

P = R.filter(pl.col("player_count") == 20)
PO = P.filter(pl.col("has_out"))

# ----------------------------------------------------------------- S1 reconcile with e2
sec("S1 reconciliation with e2 (pc20 and bomb outcome; expected 15436/499 all, 785/434 r13, 605/434 day12-25, 180/0 day01-07, 14602/41 other)")
print("all rounds, pc20, outcome: rows", PO.height, "| m1 mismatches", int(PO["m1_e2"].sum()))
r13 = PO.filter(pl.col("round") == 13)
print("round 13, pc20, outcome: rows", r13.height, "| m1", int(r13["m1_e2"].sum()))
print(r13.group_by("dayg").agg(pl.len().alias("n"), pl.col("m1_e2").sum().alias("m1")).sort("dayg"))
oth = PO.filter(pl.col("rbin") == "other")
print("other rounds, pc20, outcome: rows", oth.height, "| m1", int(oth["m1_e2"].sum()))
print("m1 by round (rounds with any m1):")
print(PO.filter(pl.col("m1_e2")).group_by("round").len().sort("round").rows())
print("pc20 rows w/ outcome, duplicate (match, round) rows present in PO:",
      PO.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height)

# ----------------------------------------------------------------- S2 build confound
sec("S2 build_num x round 13 (pc20, outcome) and build x all rounds (pc20, outcome)")
print(r13.group_by("build_num").agg(pl.len().alias("n"), pl.col("m1_e2").sum().alias("m1"),
                                   pl.col("m1_e2").mean().alias("rate")).sort("build_num"))
print(PO.group_by(["build_num", "rbin"]).agg(pl.len().alias("n"), pl.col("m1_e2").sum().alias("m1"),
                                            pl.col("m1_e2").mean().alias("rate")).sort(["build_num", "rbin"]))
print("build 9961 r13 by day (n, m1):")
print(r13.filter(pl.col("build_num") == 9961).group_by("day").agg(pl.len().alias("n"),
      pl.col("m1_e2").sum().alias("m1")).sort("day").rows())
print("build 9842 r13 by day (n, m1):")
print(r13.filter(pl.col("build_num") == 9842).group_by("day").agg(pl.len().alias("n"),
      pl.col("m1_e2").sum().alias("m1")).sort("day").rows())

# ----------------------------------------------------------------- S3 message vs mismatch (the claim)
sec("S3 win_reason_message x m1 (pc20, outcome) for round 13 -- the claim's table, recomputed")
print(r13.group_by(["m1_e2", "win_reason_message", "bomb_side", "winner_team_code"]).len()
      .sort(["m1_e2", "win_reason_message"]))
print("\nclaim check: generic share among m1 rows (r13):",
      round(float(r13.filter(pl.col("m1_e2"))["generic"].mean()), 4))
print("generic share among m1-false rows (r13):",
      round(float(r13.filter(~pl.col("m1_e2"))["generic"].mean()), 4),
      "| n =", r13.filter(~pl.col("m1_e2")).height)
sec("S3b generic share within build (ALL rounds with outcome, pc20) x m1")
print(PO.group_by(["build_num", "m1_e2"]).agg(pl.len().alias("n"),
      pl.col("generic").mean().alias("generic_share"),
      pl.col("specific").mean().alias("specific_share")).sort(["build_num", "m1_e2"]))
sec("S3c generic share within build x rbin x m1 (pc20, outcome)")
print(PO.group_by(["build_num", "rbin", "m1_e2"]).agg(pl.len().alias("n"),
      pl.col("generic").mean().alias("generic_share")).sort(["build_num", "rbin", "m1_e2"]))
sec("S3d build 9961, round 13: matched rows and their messages")
print(r13.filter((pl.col("build_num") == 9961) & (~pl.col("m1_e2"))).group_by(
      ["win_reason_message", "bomb_side"]).len())

# ----------------------------------------------------------------- S4 independent score source
sec("S4 round_state score winner vs round_end winner vs bomb outcome (pc20, outcome, score available)")
S = PO.filter(pl.col("score_winner").is_not_null())
print("rows with score winner:", S.height, "of", PO.height)
print(S.group_by(["build_num", "rbin", "m1_e2", "m2_score_vs_roundend", "m3_score_vs_bomb"]).len()
      .sort(["build_num", "rbin", "m1_e2", "m2_score_vs_roundend", "m3_score_vs_bomb"]))
sec("S4b round 13 only: score winner agreement with round_end and with bomb side")
s13 = S.filter(pl.col("round") == 13)
print(s13.group_by(["build_num", "dayg", "m1_e2", "score_winner", "winner_team_code", "bomb_side"]).len()
      .sort(["build_num", "dayg", "m1_e2", "score_winner", "winner_team_code", "bomb_side"]))

# ----------------------------------------------------------------- S5 bomb tick window
sec("S5 bomb outcome tick vs round window [round_freeze_end tick, round_end tick] (pc20, outcome)")
W = PO.filter(pl.col("fe_tick").is_not_null()).with_columns(
    pl.when(pl.col("bomb_tick") < pl.col("fe_tick")).then(pl.lit("before_freeze_end"))
      .when(pl.col("bomb_tick") > pl.col("tick")).then(pl.lit("after_round_end"))
      .otherwise(pl.lit("inside")).alias("win"),
)
print(W.group_by(["build_num", "rbin", "m1_e2", "win"]).len().sort(["build_num", "rbin", "m1_e2", "win"]))

# ----------------------------------------------------------------- S6 duplicates and multiplicity
sec("S6 round_end multiplicity (pc20) and matches")
print("pc20 rounds:", P.height, "| matches with pc20 rows:", P["match_key"].n_unique())
print("build counts (matches):", H.group_by("build_num").len().sort("build_num").rows())

print("\nDONE")
