"""verify D5-6 check 13: the no-plant wipe test with the right roster per mode.
Roster from player_spawn per match (regime A): pattern-9 matches are 2v2 (4 players), pattern-13 matches 5v5 (10 players).
Rule (no plant): a round_end T win needs every CT player dead; a CT win by elimination needs every T dead; a CT win by
time needs neither side wiped. Here: rw=T rounds with the CT side fully dead (contradiction), by group and label.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check13.py
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
sp = scan("player_spawn", ["match_key", "round", "player_id", "player_team_code"]).filter(
    pl.col("player_id").is_not_null()).collect()
roster = sp.group_by("match_key").agg(pl.col("player_id").n_unique().alias("players_in_match"))
MP = M.filter(pl.col("r1_end")).select(["match_key", "grp", "map_name"]).join(
    pat.select(["match_key", "pattern"]), on="match_key", how="left").join(roster, on="match_key", how="left").with_columns(
    pattern=pl.col("pattern").fill_null("none"))
MP = MP.with_columns(pg=pl.when(pl.col("pattern") == "9").then(pl.lit("p9"))
                     .when(pl.col("pattern").str.starts_with("13")).then(pl.lit("p13"))
                     .when(pl.col("pattern") == "none").then(pl.lit("none")).otherwise(pl.lit("other")),
                     per_side=pl.when(pl.col("players_in_match") >= 9).then(pl.lit(5)).otherwise(pl.lit(2)))
print("roster per side by group:")
print(MP.group_by(["pg", "per_side"]).agg(pl.len().alias("matches")).sort(["pg", "per_side"]))

DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).filter(
    pl.col("player_team_code").is_in([2, 3]) & pl.col("player_id").is_not_null()).collect()
NP = (T.filter(pl.col("re_present") & pl.col("rw").is_not_null() & (pl.col("bs_planted") == 0)).join(
    MP.select(["match_key", "grp", "map_name", "pg", "per_side"]), on="match_key", how="left").filter(
    (pl.col("grp") == "ctrl_late") & (pl.col("map_name") != "cs_office")).select(
    K + ["rw", "pg", "per_side", "fe_tick", "re_tick"]))
D = DE.join(NP.select(K + ["fe_tick", "re_tick"]), on=K, how="inner").filter(
    (pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
ps = (D.group_by(K + ["player_team_code"]).agg(pl.col("player_id").n_unique().alias("n"))
      .pivot(on="player_team_code", index=K, values="n", aggregate_function="first")
      .rename({"2": "T_dead", "3": "CT_dead"}))
W = NP.join(ps, on=K, how="left").with_columns(pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
W = W.filter((pl.col("T_dead") + pl.col("CT_dead")) > 0)
W = W.with_columns(CTfull=pl.col("CT_dead") >= pl.col("per_side"), Tfull=pl.col("T_dead") >= pl.col("per_side"))
print("\n== M1: no-plant rounds with deaths (ctrl late regime A, cs_office excluded), by group and label ==")
print("rw=T: share with CT side fully dead (contradiction if high-false). rw=CT: share with CT side fully dead (contradiction).")
print(W.filter(pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 25, 28])).group_by(["pg", "round", "rw"]).agg(
    pl.len().alias("n"), pl.col("CTfull").mean().round(3).alias("CT_full_share"),
    pl.col("Tfull").mean().round(3).alias("T_full_share")).sort(["pg", "round", "rw"]))
print("\n== M2: label 9 in pattern-9 (2v2) matches, rw=T rows: CT both dead, and T both dead ==")
s9 = W.filter((pl.col("round") == 9) & (pl.col("pg") == "p9") & (pl.col("rw") == 2))
print("rw=T rows:", s9.height, " CT full:", int(s9["CTfull"].sum()), " T full:", int(s9["Tfull"].sum()))
print("\n== M3: label 9 in pattern-9 matches, rw=CT rows: CT full (contradiction) and T full ==")
s9c = W.filter((pl.col("round") == 9) & (pl.col("pg") == "p9") & (pl.col("rw") == 3))
print("rw=CT rows:", s9c.height, " CT full:", int(s9c["CTfull"].sum()), " T full:", int(s9c["Tfull"].sum()))
