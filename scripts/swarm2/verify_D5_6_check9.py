"""verify D5-6 check 9: pattern-9 matches, label structure. Are their death rows inside their own round windows, and do
full wipes (a whole side dead) occur at labels 8, 10, 11, 12, 14 in those matches? Plus a 2-match timeline for labels 7-11.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check9.py
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
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
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
TT = T.filter(pl.col("re_present") & pl.col("rw").is_not_null()).select(K + ["rw", "chain", "fe_tick", "re_tick", "bs_planted"])
TT = TT.with_columns(chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
                     flag=pl.when(pl.col("chain").is_not_null() & (pl.col("chain") != 0))
                     .then((pl.col("chain") != pl.col("rw")).cast(pl.Int8)).otherwise(None))
A = TT.join(M.filter(pl.col("r1_end")).select(["match_key"]), on="match_key", how="inner")
pat = (A.filter(pl.col("flag") == 1).group_by("match_key").agg(pl.col("round").sort().alias("fl")))
pat = pat.with_columns(pattern=pl.col("fl").list.eval(pl.element().cast(pl.Utf8)).list.join(","))
MP = M.filter(pl.col("r1_end")).select(["match_key", "grp", "map_name"]).join(
    pat.select(["match_key", "pattern"]), on="match_key", how="left").with_columns(pattern=pl.col("pattern").fill_null("none"))
p9 = MP.filter(pl.col("pattern") == "9").select("match_key")
p13 = MP.filter(pl.col("pattern").str.starts_with("13")).select("match_key")

DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).filter(
    pl.col("player_team_code").is_in([2, 3])).collect()
DW = DE.join(T.select(K + ["fe_tick", "re_tick"]), on=K, how="left").with_columns(
    inwin=(pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
DW = DW.join(MP.select(["match_key", "pattern"]), on="match_key", how="inner")
print("== I1: death window share [fe_r, et_r] by pattern group and label (regime A late) ==")
print(DW.filter(pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14])).with_columns(
    pg=pl.when(pl.col("pattern") == "9").then(pl.lit("p9")).when(pl.col("pattern").str.starts_with("13")).then(pl.lit("p13"))
    .otherwise(pl.lit("other"))).group_by(["pg", "round"]).agg(
    pl.len().alias("n"), pl.col("inwin").mean().round(3).alias("share_inwin")).sort(["pg", "round"]))

# wipes: per no-plant round with deaths, T_dead / CT_dead (distinct players) inside the window
NP = T.filter(pl.col("re_present") & pl.col("rw").is_not_null() & (pl.col("bs_planted") == 0)).join(
    M.select(["match_key", "grp"]), on="match_key", how="left").filter(pl.col("grp") == "ctrl_late").select(
    K + ["rw", "fe_tick", "re_tick"])
Dn = DE.filter(pl.col("player_id").is_not_null()).join(NP.select(K + ["fe_tick", "re_tick"]), on=K, how="inner").filter(
    (pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
ps = (Dn.group_by(K + ["player_team_code"]).agg(pl.col("player_id").n_unique().alias("n"))
      .pivot(on="player_team_code", index=K, values="n", aggregate_function="first")
      .rename({"2": "T_dead", "3": "CT_dead"}))
W = NP.join(ps, on=K, how="left").with_columns(pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
W = W.join(MP.select(["match_key", "pattern"]), on="match_key", how="left")
W = W.with_columns(pg=pl.when(pl.col("pattern") == "9").then(pl.lit("p9"))
                   .when(pl.col("pattern").str.starts_with("13")).then(pl.lit("p13")).otherwise(pl.lit("other")))
print("\n== I2: no-plant rounds with deaths: share with T all dead, CT all dead, neither, by pattern and label ==")
print(W.filter(pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14])).group_by(["pg", "round"]).agg(
    pl.len().alias("n"), (pl.col("T_dead") >= 5).mean().round(3).alias("T_all5"),
    (pl.col("CT_dead") >= 5).mean().round(3).alias("CT_all5"),
    ((pl.col("T_dead") < 5) & (pl.col("CT_dead") < 5)).mean().round(3).alias("neither"),
    pl.col("T_dead").mean().round(2).alias("mean_T_dead"), pl.col("CT_dead").mean().round(2).alias("mean_CT_dead")).sort(["pg", "round"]))

# victims per round in p9 vs p13 (all labels, regime A late), to see whether p9 rounds have full deaths at all
print("\n== I3: victims per round (all rounds with deaths, regime A late), by pattern group ==")
vr = (DW.group_by(["pattern", "match_key", "round"]).agg(pl.len().alias("nd")))
vr = vr.with_columns(pg=pl.when(pl.col("pattern") == "9").then(pl.lit("p9"))
                     .when(pl.col("pattern").str.starts_with("13")).then(pl.lit("p13")).otherwise(pl.lit("other")))
print(vr.group_by("pg").agg(pl.len().alias("rounds_with_deaths"), pl.col("nd").mean().round(2).alias("mean_victims_per_round"),
                             pl.col("nd").median().alias("median_victims")).sort("pg"))

# two p9 timelines (first two by match_key), labels 7-11
mkeys = p9["match_key"].unique().sort().head(2).to_list()
print("\n== I4: timeline, two pattern-9 matches, labels 7-11 ==")
tl = T.filter(pl.col("match_key").is_in(mkeys) & pl.col("round").is_between(7, 11)).select(
    ["match_key", "round", "rw", "chain", "implied", "fe_tick", "re_tick", "bs_planted", "bs_defused", "bs_exploded"])
print(tl.sort(["match_key", "round"]))
