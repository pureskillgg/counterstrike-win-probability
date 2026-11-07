"""verify D5-6 check 12: players per round from an independent channel (player_spawn), by flag group (regime A).
If pattern-9 matches are 2v2, distinct players per round is ~4; if 5v5 with incomplete death rows, it is ~10.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check12.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(40)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_6")
K = ["match_key", "round"]


def scan(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.scan_parquet(fs), fs


M = pl.read_parquet(SCR / "v_matches.parquet")
T = pl.read_parquet(SCR / "v_rounds.parquet")
LATE = [f"{d:02d}" for d in range(12, 26)]
M = M.with_columns(grp=pl.when(pl.col("aff_pc20")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late")).otherwise(pl.lit("other")))
TT = T.filter(pl.col("re_present") & pl.col("rw").is_not_null()).select(K + ["rw", "chain"])
TT = TT.with_columns(chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
                     flag=pl.when(pl.col("chain").is_not_null() & (pl.col("chain") != 0))
                     .then((pl.col("chain") != pl.col("rw")).cast(pl.Int8)).otherwise(None))
A = TT.join(M.filter(pl.col("r1_end")).select(["match_key"]), on="match_key", how="inner")
pat = (A.filter(pl.col("flag") == 1).group_by("match_key").agg(pl.col("round").sort().alias("fl")))
pat = pat.with_columns(pattern=pl.col("fl").list.eval(pl.element().cast(pl.Utf8)).list.join(","))
MP = M.filter(pl.col("r1_end")).select(["match_key", "grp"]).join(
    pat.select(["match_key", "pattern"]), on="match_key", how="left").with_columns(pattern=pl.col("pattern").fill_null("none"))
MP = MP.with_columns(pg=pl.when(pl.col("pattern") == "9").then(pl.lit("p9"))
                     .when(pl.col("pattern").str.starts_with("13")).then(pl.lit("p13"))
                     .when(pl.col("pattern") == "none").then(pl.lit("none")).otherwise(pl.lit("other")))

lf, fs = scan("player_spawn")
print("player_spawn files:", len(fs), " schema:", {k: str(v) for k, v in lf.collect_schema().items()})
sp = lf.select(["match_key", "round", "player_id", "player_team_code"]).filter(
    pl.col("player_id").is_not_null()).collect()
per_round = sp.group_by(K).agg(pl.col("player_id").n_unique().alias("n_players"),
                               pl.col("player_team_code").n_unique().alias("n_team_codes"))
per_round = per_round.join(MP.select(["match_key", "pg"]), on="match_key", how="inner")
print("\n== L1: distinct spawned players per round, regime A, by flag group ==")
print(per_round.group_by("pg").agg(pl.len().alias("rounds"), pl.col("n_players").median().alias("median_players"),
                                   pl.col("n_players").mean().round(2).alias("mean_players")).sort("pg"))
print(per_round.group_by(["pg", "n_players"]).agg(pl.len().alias("rounds")).sort(["pg", "n_players"]))

pm = sp.group_by("match_key").agg(pl.col("player_id").n_unique().alias("players_in_match")).join(
    MP.select(["match_key", "pg"]), on="match_key", how="inner")
print("\n== L2: distinct spawned players per match, regime A, by flag group ==")
print(pm.group_by(["pg", "players_in_match"]).agg(pl.len().alias("matches")).sort(["pg", "players_in_match"]))
