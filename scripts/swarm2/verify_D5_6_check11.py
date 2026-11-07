"""verify D5-6 check 11: roster size per side per match (distinct player_id on each side in player_death), by flag group,
and a debug of round_end player_count (rw_pc) against the flag groups.
Roster proxy (ASSUMPTION): distinct victim player_id per side per match, max over the match.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check11.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_6")
K = ["match_key", "round"]


def scan(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.scan_parquet(fs).select(cols)


T = pl.read_parquet(SCR / "v_rounds.parquet")
M = pl.read_parquet(SCR / "v_matches.parquet")
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
MP = M.filter(pl.col("r1_end")).select(["match_key", "grp", "map_name"]).join(
    pat.select(["match_key", "pattern"]), on="match_key", how="left").with_columns(pattern=pl.col("pattern").fill_null("none"))
MP = MP.with_columns(pg=pl.when(pl.col("pattern") == "9").then(pl.lit("p9"))
                     .when(pl.col("pattern").str.starts_with("13")).then(pl.lit("p13"))
                     .when(pl.col("pattern") == "none").then(pl.lit("none")).otherwise(pl.lit("other")))

DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).filter(
    pl.col("player_team_code").is_in([2, 3]) & pl.col("player_id").is_not_null()).collect()
roster = (DE.group_by(["match_key", "player_team_code", "player_id"]).agg(pl.len().alias("n")).group_by(
    ["match_key", "player_team_code"]).agg(pl.len().alias("distinct_players")))
rost = roster.pivot(on="player_team_code", index="match_key", values="distinct_players", aggregate_function="first").rename(
    {"2": "T_roster", "3": "CT_roster"})
R = MP.join(rost, on="match_key", how="left")
print("== K1: distinct victim players per side per match, by flag group (regime A, all matches) ==")
print(R.group_by("pg").agg(pl.len().alias("matches"), pl.col("T_roster").median().alias("med_T"),
                           pl.col("CT_roster").median().alias("med_CT"),
                           pl.col("T_roster").mean().round(2).alias("mean_T"),
                           pl.col("CT_roster").mean().round(2).alias("mean_CT")).sort("pg"))
print("histogram of CT_roster by group:")
print(R.group_by(["pg", "CT_roster"]).agg(pl.len().alias("n")).sort(["pg", "CT_roster"]))

# rw_pc debug: per-round player_count vs flag group
RE = scan("round_end", ["match_key", "round", "player_count"]).collect()
rpc = RE.group_by("match_key").agg(pl.col("player_count").n_unique().alias("n_pc_vals"),
                                   pl.col("player_count").min().alias("min_pc"), pl.col("player_count").max().alias("max_pc"))
print("\n== K2: round_end player_count per match, by flag group ==")
print(MP.join(rpc, on="match_key", how="left").group_by("pg").agg(
    pl.len().alias("matches"), pl.col("min_pc").value_counts().alias("min_pc_counts")).sort("pg") if False else
    MP.join(rpc, on="match_key", how="left").group_by(["pg", "min_pc", "max_pc"]).agg(pl.len().alias("matches")).sort(["pg", "matches"], descending=[False, True]).head(30))
