"""verify D5-6 check 10: game mode. Does the flag pattern follow round_end player_count (8 = 2v2, 20 = 5v5)?
Then redo the no-plant wipe test with the correct number of players per side for each mode:
  5v5 (player_count 20): a T win without a plant needs all 5 CT dead; 2v2 (player_count 8): needs both CT dead.
Also restricts the label-13 headline to 5v5 rounds.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check10.py
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
TT = T.filter(pl.col("re_present") & pl.col("rw").is_not_null()).select(K + ["rw", "rw_pc", "chain", "bs_planted", "fe_tick", "re_tick"])
TT = TT.with_columns(chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
                     flag=pl.when(pl.col("chain").is_not_null() & (pl.col("chain") != 0))
                     .then((pl.col("chain") != pl.col("rw")).cast(pl.Int8)).otherwise(None))
A = TT.join(M.filter(pl.col("r1_end")).select(["match_key"]), on="match_key", how="inner")
pat = (A.filter(pl.col("flag") == 1).group_by("match_key").agg(pl.col("round").sort().alias("fl")))
pat = pat.with_columns(pattern=pl.col("fl").list.eval(pl.element().cast(pl.Utf8)).list.join(","))
mpc = (T.filter(pl.col("re_present") & pl.col("rw_pc").is_not_null()).group_by("match_key")
       .agg(pl.col("rw_pc").mode().first().alias("mode_pc"), pl.col("rw_pc").n_unique().alias("n_pc_values")))
MP = M.filter(pl.col("r1_end")).select(["match_key", "grp", "map_name"]).join(
    pat.select(["match_key", "pattern"]), on="match_key", how="left").join(mpc, on="match_key", how="left").with_columns(
    pattern=pl.col("pattern").fill_null("none"))
MP = MP.with_columns(pg=pl.when(pl.col("pattern") == "9").then(pl.lit("p9"))
                     .when(pl.col("pattern").str.starts_with("13")).then(pl.lit("p13"))
                     .when(pl.col("pattern") == "none").then(pl.lit("none")).otherwise(pl.lit("other")))

print("== J1: flag pattern group x modal round_end player_count (regime A matches) ==")
print(MP.group_by(["pg", "mode_pc"]).agg(pl.len().alias("matches")).sort(["pg", "matches"], descending=[False, True]))
print("matches whose round_end player_count varies within the match:", int((MP["n_pc_values"] > 1).sum()))

DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).filter(
    pl.col("player_team_code").is_in([2, 3]) & pl.col("player_id").is_not_null()).collect()
NP = (T.filter(pl.col("re_present") & pl.col("rw").is_not_null() & (pl.col("bs_planted") == 0)).join(
    M.select(["match_key", "grp", "map_name"]), on="match_key", how="left").filter(
    (pl.col("grp") == "ctrl_late") & (pl.col("map_name") != "cs_office")).select(
    K + ["rw", "rw_pc", "fe_tick", "re_tick"]).join(MP.select(["match_key", "pg"]), on="match_key", how="left"))
Dn = DE.join(NP.select(K + ["fe_tick", "re_tick"]), on=K, how="inner").filter(
    (pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
ps = (Dn.group_by(K + ["player_team_code"]).agg(pl.col("player_id").n_unique().alias("n"))
      .pivot(on="player_team_code", index=K, values="n", aggregate_function="first")
      .rename({"2": "T_dead", "3": "CT_dead"}))
W = NP.join(ps, on=K, how="left").with_columns(pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
W = W.filter((pl.col("T_dead") + pl.col("CT_dead")) > 0)
W = W.with_columns(
    need=pl.when(pl.col("rw_pc") == 20).then(pl.lit(5)).when(pl.col("rw_pc") == 8).then(pl.lit(2)).otherwise(None))
print("\n== J2: no-plant rounds (ctrl late, regime A, cs_office excluded), rw=T: share with CT fully dead (needed count by mode) ==")
print(W.filter(pl.col("rw") == 2).filter(pl.col("need").is_not_null()).with_columns(
    CTfull=pl.col("CT_dead") >= pl.col("need"), Tfull=pl.col("T_dead") >= pl.col("need")).group_by(
    ["rw_pc", "round"]).agg(pl.len().alias("n_rwT"), pl.col("CTfull").sum().alias("CT_full"),
                            pl.col("CTfull").mean().round(3).alias("share_CT_full")).sort(["rw_pc", "round"]).filter(
    pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 25, 28])))
print("\n== J3: label 13 headline restricted to 5v5 (rw_pc == 20), no-plant, ctrl late, regime A, cs_office excluded ==")
h = W.filter((pl.col("round") == 13) & (pl.col("rw_pc") == 20))
print("rows:", h.height, " rw=T:", int((h["rw"] == 2).sum()), " rw=T with CT all 5 dead:",
      int(((h["rw"] == 2) & (h["CT_dead"] >= 5)).sum()), " rw=CT with CT all 5 dead:",
      int(((h["rw"] == 3) & (h["CT_dead"] >= 5)).sum()))
print("label 12 and 14 5v5 rw=T CT all 5 dead (baseline):")
for lab in [12, 14]:
    s = W.filter((pl.col("round") == lab) & (pl.col("rw_pc") == 20) & (pl.col("rw") == 2))
    print(f"label {lab}: rw=T rows {s.height}, CT all 5 dead {int((s['CT_dead'] >= 5).sum())}")
print("\n== J4: label 9 in 2v2 matches (rw_pc == 8, pattern 9) vs 2v2 without flag: rw=T with CT both dead ==")
for g in ["p9", "other", "none"]:
    s = W.filter((pl.col("round") == 9) & (pl.col("rw_pc") == 8) & (pl.col("pg") == g) & (pl.col("rw") == 2))
    print(f"pg={g}: rw=T rows {s.height}, CT both dead {int((s['CT_dead'] >= 2).sum())}")
print("label 8 and 10, 2v2 pattern-9 matches, rw=T: CT both dead (baseline):")
for lab in [8, 10]:
    s = W.filter((pl.col("round") == lab) & (pl.col("rw_pc") == 8) & (pl.col("pg") == "p9") & (pl.col("rw") == 2))
    print(f"label {lab}: rw=T rows {s.height}, CT both dead {int((s['CT_dead'] >= 2).sum())}")
