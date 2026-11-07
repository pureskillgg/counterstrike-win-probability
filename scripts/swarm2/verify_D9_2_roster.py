"""verify_D9_2 roster: where do player team codes flip (true halftime) per build; roster vs bomb-event team codes.

Reads player_info (round, player_id, team_code), bomb_state planter/defuser rows, and the core scratch parquet.
Team codes: 2 = T, 3 = CT (as used in the claim). For each match and each label r, the number of players whose
team_code at label r differs from label r-1 is counted; the label with the most flips is the halftime position.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


R = pl.read_parquet(SCR / "verify_D9_2_round.parquet")
H = pl.scan_parquet(files_of(BASE / "header")).select(["key", "build_num"]).collect().with_columns(
    pl.col("key").alias("match_key"))
PI = pl.scan_parquet(files_of(BASE / f"basic_player_info{SUF}")).select(
    ["match_key", "round", "player_id", "team_code"]).collect()
print("player_info rows:", PI.height)
PI = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
print("player_info unique (match,round,player):", PI.height)
PI = PI.join(H.select(["match_key", "build_num"]), on="match_key", how="left")
print("team_code values:", PI.group_by("team_code").len().sort("team_code").rows())
print("players per (match,round) distribution (top):",
      PI.group_by(["match_key", "round"]).agg(pl.len().alias("n_players")).group_by("n_players")
        .agg(pl.len().alias("n_rounds")).sort("n_players").rows())

# ---- switch detection: for each player and round r, did team_code change from r-1 to r?
prev = (PI.with_columns((pl.col("round") + 1).alias("round_next"))
          .select(["match_key", "round_next", "player_id", "team_code"])
          .rename({"round_next": "round", "team_code": "tc_prev"}))
J = PI.join(prev, on=["match_key", "round", "player_id"], how="inner")
J = J.with_columns((pl.col("team_code") != pl.col("tc_prev")).alias("flip"))
flips = (J.group_by(["build_num", "round"]).agg(pl.len().alias("pairs"), pl.col("flip").sum().alias("flips"))
          .with_columns((pl.col("flips") / pl.col("pairs")).round(3).alias("flip_rate"))
          .sort(["build_num", "round"]))
print("\n== per build, flip rate of team_code from round r-1 to r (labels 1..30)")
for b in [9842, 9961]:
    sub = flips.filter(pl.col("build_num") == b).filter(pl.col("round") <= 30)
    print(f"-- build {b}:")
    print(sub.select(["round", "pairs", "flips", "flip_rate"]))

# ---- match-level: label of the largest flip rate, per build
mf = (J.group_by(["match_key", "build_num", "round"]).agg(pl.col("flip").mean().alias("fr")))
top = (mf.sort(["match_key", "fr", "round"], descending=[False, True, False])
         .group_by("match_key").agg(pl.col("round").first().alias("flip_round"),
                                    pl.col("fr").first().alias("top_fr"),
                                    pl.col("build_num").first()))
print("\n== per build: distribution of the label with the largest team-code flip, per match")
print(top.group_by(["build_num", "flip_round"]).len().sort(["build_num", "flip_round"]))

# ---- roster vs bomb-event team codes on bomb-outcome rounds (pc20)
core = R.filter((pl.col("player_count") == 20) & pl.col("has_out"))
rp = (core.join(PI.select(["match_key", "round", "player_id", "team_code"]).rename({"player_id": "planter_id",
                                                                                   "team_code": "planter_roster"}),
                on=["match_key", "round", "planter_id"], how="left")
           .join(PI.select(["match_key", "round", "player_id", "team_code"]).rename({"player_id": "defuser_id",
                                                                                    "team_code": "defuser_roster"}),
                 on=["match_key", "round", "defuser_id"], how="left"))
print("\n== bomb rounds (pc20): planter roster team code and event planter team code, by build and mm")
print(rp.group_by(["build_num", "round", "mm", "planter_team", "planter_roster"]).len()
        .filter(pl.col("round").is_in([9, 13, 25, 28, 5, 12, 14]))
        .sort(["build_num", "round", "mm"]))
print("\n== bomb rounds: defuser roster vs defuser event team code, by build")
print(rp.filter(pl.col("defuser_team").is_not_null())
        .group_by(["build_num", "mm", "defuser_team", "defuser_roster"]).len().sort(["build_num", "mm"]))
rp.write_parquet(SCR / "verify_D9_2_roster_bomb.parquet")
print("\nDONE roster")
