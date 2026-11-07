"""D9 probe 8: per-round-index death-consistency of the recorded round_end winner, by build (non-bomb rounds).

Consistency (ASSUMPTION, same rule as D9_tests4/5): winner 2 needs CT_dead >= n_CT; winner 3 needs T_dead >= n_T.
Only rounds with no bomb outcome and a player_info roster (n_T, n_CT not null) are scored.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9")
R = pl.read_parquet(SCR / "D9_round.parquet")
M = pl.read_parquet(SCR / "D9_match.parquet")


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want):
    fd = BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in want if c in have]).collect()


DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead")))
BS = scan("bomb_state", ["match_key", "round", "event_type"])
bomb_rounds = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("bomb_any")))
T = (R.join(M.select(["match_key", "build_num"]), on="match_key", how="left")
       .join(dead, on=["match_key", "round"], how="left")
       .join(bomb_rounds, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_any").fill_null(False), pl.col("T_dead").fill_null(0),
                     pl.col("CT_dead").fill_null(0))
       .filter(~pl.col("bomb_any") & pl.col("n_CT").is_not_null() & (pl.col("round") <= 40)))
T = T.with_columns(
    pl.when(pl.col("winner_team_code") == 2).then(pl.col("CT_dead") >= pl.col("n_CT"))
      .otherwise(pl.col("T_dead") >= pl.col("n_T")).alias("ok"))
out = (T.group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("ok").mean().round(2).alias("rate"))
       .sort(["build_num", "round"]))
for b in [9842, 9961]:
    g = out.filter(pl.col("build_num") == b)
    print(f"== build {b}: round index -> (n, consistency rate)")
    print("   " + "  ".join(f"r{r}:({n},{rt})" for r, n, rt in g.select(["round", "n", "rate"]).rows()))
