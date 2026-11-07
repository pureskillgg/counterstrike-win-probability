"""verify D5-6 check 7: is label 9 the same inversion, seen inside the matches that flag label 9?

Within regime A ctrl-late matches (cs_office excluded), no-plant rounds: among rw=T rounds, the share with CT fully dead,
split by whether the match's flag pattern is "9" (chain disagrees at label 9 and nowhere else) or not.
Expected under the first-post-swap-round inversion: ~0 in pattern-9 matches at label 9, baseline (~50%) elsewhere.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check7.py
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
TT = T.filter(pl.col("re_present") & pl.col("rw").is_not_null()).select(K + ["rw", "chain", "bs_planted"])
TT = TT.with_columns(chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
                     flag=pl.when(pl.col("chain").is_not_null() & (pl.col("chain") != 0))
                     .then((pl.col("chain") != pl.col("rw")).cast(pl.Int8)).otherwise(None))
A = TT.join(M.filter(pl.col("r1_end")).select(["match_key", "map_name"]), on="match_key", how="inner")
pat = (A.filter(pl.col("flag") == 1).group_by("match_key").agg(pl.col("round").sort().alias("fl")))
pat = pat.with_columns(pattern=pl.col("fl").list.eval(pl.element().cast(pl.Utf8)).list.join(","))
MP = M.filter(pl.col("r1_end")).select(["match_key", "map_name", "grp"]).join(
    pat.select(["match_key", "pattern"]), on="match_key", how="left").with_columns(pattern=pl.col("pattern").fill_null("none"))

DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).filter(
    pl.col("player_team_code").is_in([2, 3]) & pl.col("player_id").is_not_null()).collect()
NP = (T.filter(pl.col("re_present") & pl.col("rw").is_not_null() & (pl.col("bs_planted") == 0))
      .join(MP, on="match_key", how="left")
      .filter((pl.col("grp") == "ctrl_late") & (pl.col("map_name") != "cs_office"))
      .select(K + ["rw", "pattern", "fe_tick", "re_tick", "chain"]))
D = DE.join(NP.select(K + ["fe_tick", "re_tick"]), on=K, how="inner").filter(
    (pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
ps = (D.group_by(K + ["player_team_code"]).agg(pl.col("player_id").n_unique().alias("n"))
      .pivot(on="player_team_code", index=K, values="n", aggregate_function="first")
      .rename({"2": "T_dead", "3": "CT_dead"}))
X = NP.join(ps, on=K, how="left").with_columns(pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
X = X.filter((pl.col("T_dead") + pl.col("CT_dead")) > 0)
X = X.with_columns(in9=(pl.col("pattern") == "9"), in13=(pl.col("pattern").str.starts_with("13")))

print("== G1: label 9, no-plant, ctrl late regime A (cs_office excluded), rw=T rows: CT all dead by match pattern ==")
g1 = X.filter((pl.col("round") == 9) & (pl.col("rw") == 2)).group_by("in9").agg(
    pl.len().alias("n_rwT"), (pl.col("CT_dead") >= 5).sum().alias("CT_all5"),
    (pl.col("CT_dead") >= 5).mean().round(3).alias("share")).sort("in9")
print(g1)

print("\n== G2: label 13, no-plant, ctrl late regime A (cs_office excluded), rw=T rows: CT all dead by match pattern ==")
g2 = X.filter((pl.col("round") == 13) & (pl.col("rw") == 2)).group_by("in13").agg(
    pl.len().alias("n_rwT"), (pl.col("CT_dead") >= 5).sum().alias("CT_all5"),
    (pl.col("CT_dead") >= 5).mean().round(3).alias("share")).sort("in13")
print(g2)

print("\n== G3: label 9 rw=CT rows with CT all dead (contradiction), by pattern-9 membership ==")
print(X.filter((pl.col("round") == 9) & (pl.col("rw") == 3)).group_by("in9").agg(
    pl.len().alias("n_rwCT"), (pl.col("CT_dead") >= 5).sum().alias("CT_all5_contradiction")).sort("in9"))
