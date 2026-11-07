"""D5-8 verifier extract: per (match, round) table rebuilt from raw 2024 tome channels (projected lazy scans).

Fields (all from the tome, not from the claimed scratchpad tables):
  round_end        : winner_team_code, win_reason_code, player_count, tick (dedup by match, round)
  round_state      : freeze_end t_score/ct_score/phase (first row per match, round);
                     regime marker = any round_announce_match_start row in the match
  bomb_state       : exploded / defused presence per match, round; planter player_team_code
Output: scratchpad D5_8v/rounds_v.parquet, D5_8v/matches_v.parquet (scratch, outside the repo).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_extract.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_8v")
OUT.mkdir(parents=True, exist_ok=True)


def scan(ch):
    fd = os.path.join(BASE, f"basic_{ch}{SUF}")
    return pl.scan_parquet(sorted(glob.glob(os.path.join(fd, "dataframe_*"))), missing_columns="insert",
                           extra_columns="ignore")


RE = (scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
      .group_by(["match_key", "round"]).agg(pl.col("winner_team_code").first().alias("winner"),
                                            pl.col("win_reason_code").first().alias("reason"),
                                            pl.col("player_count").first().alias("pc"),
                                            pl.col("tick").min().alias("end_tick"),
                                            pl.len().alias("re_rows"),
                                            pl.col("winner_team_code").n_unique().alias("re_nwin"))
      .collect())

RS = scan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"]).collect()
REG = (RS.filter(pl.col("event_type") == "round_announce_match_start")
       .group_by("match_key").agg(pl.len().alias("n_mstart")))
FE = (RS.filter(pl.col("event_type") == "round_freeze_end")
      .sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("fe_t"),
                                            pl.col("ct_score").first().alias("fe_ct"),
                                            pl.col("phase").first().alias("fe_phase"),
                                            pl.col("tick").min().alias("fe_tick"),
                                            pl.len().alias("fe_rows")))
FE_RPM = FE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).select(pl.col("n").max().alias("max_fe_per_round"))
print("freeze_end max rows per (match, round):", int(FE_RPM.item()))
del RS

BS = scan("bomb_state").select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
BO = (BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("bevs")))
PL = (BS.filter(pl.col("event_type") == "bomb_planted").sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter"),
                                            pl.len().alias("n_plant_rows")))
del BS

R = (RE.join(FE, on=["match_key", "round"], how="left")
     .join(BO, on=["match_key", "round"], how="left")
     .join(PL, on=["match_key", "round"], how="left")
     .join(REG, on="match_key", how="left")
     .with_columns(
         day=pl.col("match_key").str.split("/").list.get(3),
         regime=pl.when(pl.col("n_mstart").fill_null(0) > 0).then(pl.lit("A")).otherwise(pl.lit("B")),
         bomb_exp=pl.col("bevs").list.contains("bomb_exploded").fill_null(False),
         bomb_def=pl.col("bevs").list.contains("bomb_defused").fill_null(False))
     .sort(["match_key", "round"]))
R.write_parquet(OUT / "rounds_v.parquet")

M = (R.group_by("match_key").agg(pl.col("day").first(), pl.col("regime").first(),
                                 pl.col("round").max().alias("max_round"),
                                 pl.col("round").n_unique().alias("n_rounds"))
     .sort("match_key"))
M.write_parquet(OUT / "matches_v.parquet")

print("rounds:", R.height, " matches:", M.height)
print("regime counts (matches):", M.group_by("regime").len().sort("regime").rows())
print("re_rows distribution:", R.group_by("re_rows").len().sort("re_rows").rows())
print("re_nwin>1 pairs:", int(R.filter(pl.col("re_nwin") > 1).height))
print("rounds with no freeze_end row:", int(R.filter(pl.col("fe_t").is_null()).height))
print("rounds with bomb outcome:", int(R.filter(pl.col("bomb_exp") | pl.col("bomb_def")).height))
print("winner value counts:", R.group_by("winner").len().sort("winner").rows())
print("reason value counts:", R.group_by("reason").len().sort("reason").rows())
print("phase at freeze_end:", R.group_by("fe_phase").len().sort("fe_phase").rows())
print("day counts (matches):", M.group_by("day").len().sort("day").rows())
