"""verify D5-6 check 8: is the pattern-9 / pattern-13 contradiction an artefact of missing death rows in those matches?

(a) death-row completeness per flag group: victim rows per round by side, and null-side share.
(b) bomb outcome at the flagged halftime label (bomb_state defused/exploded, independent of deaths and side codes):
    crosstab bomb side x rw x chain, for pattern-9 matches at label 9 and pattern-13 matches at label 13.
(c) joint (rw, T_dead, CT_dead) for no-plant rounds at the flagged label, pattern groups.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check8.py
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
TT = T.filter(pl.col("re_present") & pl.col("rw").is_not_null()).select(K + ["rw", "chain", "bs_planted", "bomb_side", "fe_tick", "re_tick"])
TT = TT.with_columns(chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
                     flag=pl.when(pl.col("chain").is_not_null() & (pl.col("chain") != 0))
                     .then((pl.col("chain") != pl.col("rw")).cast(pl.Int8)).otherwise(None))
A = TT.join(M.filter(pl.col("r1_end")).select(["match_key", "map_name"]), on="match_key", how="inner")
pat = (A.filter(pl.col("flag") == 1).group_by("match_key").agg(pl.col("round").sort().alias("fl")))
pat = pat.with_columns(pattern=pl.col("fl").list.eval(pl.element().cast(pl.Utf8)).list.join(","))
MP = M.filter(pl.col("r1_end")).select(["match_key", "grp"]).join(
    pat.select(["match_key", "pattern"]), on="match_key", how="left").with_columns(
    pattern=pl.col("pattern").fill_null("none"))
MP = MP.with_columns(pgrp=pl.when(pl.col("pattern").str.starts_with("13")).then(pl.lit("p13"))
                     .when(pl.col("pattern") == "9").then(pl.lit("p9"))
                     .when(pl.col("pattern") == "none").then(pl.lit("none")).otherwise(pl.lit("other")))

# (a) death completeness per flag group, all rounds of regime A ctrl late
DALL = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).collect()
DALL = DALL.join(MP.select(["match_key", "pgrp", "grp"]), on="match_key", how="left").filter(pl.col("grp") == "ctrl_late")
rounds_per = T.filter(pl.col("re_present")).join(MP.select(["match_key", "pgrp", "grp"]), on="match_key", how="left").filter(
    pl.col("grp") == "ctrl_late").group_by("pgrp").agg(pl.len().alias("rounds"))
side = DALL.group_by(["pgrp", "player_team_code"], maintain_order=False).agg(pl.len().alias("n")).sort(["pgrp", "player_team_code"])
print("== H1 (a): death rows by side per flag group, ctrl late regime A ==")
print(side)
print(rounds_per.sort("pgrp"))
print("null-side share by group:")
print(DALL.group_by("pgrp").agg(pl.len().alias("n"), pl.col("player_team_code").is_null().mean().round(4).alias("null_side_share")).sort("pgrp"))

# (b) bomb outcome at the flagged halftime label, independent check
BO = (scan("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect()
      .group_by(K).agg(pl.col("event_type").unique().alias("evs"))
      .with_columns(bside=pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)))
      .select(K + ["bside"]))
B = TT.join(MP.select(["match_key", "pattern", "pgrp", "grp"]), on="match_key", how="inner").join(BO, on=K, how="inner")
print("\n== H2 (b): bomb outcome vs rw vs chain at label 9 (pattern 9) and label 13 (pattern 13), ctrl late regime A ==")
b9 = B.filter((pl.col("pgrp") == "p9") & (pl.col("round") == 9) & (pl.col("grp") == "ctrl_late"))
print("pattern-9 label 9 bomb rows:", b9.height)
print(b9.group_by(["bside", "rw", "chain"]).agg(pl.len().alias("n")).sort(["bside", "rw", "chain"]))
b13 = B.filter((pl.col("pgrp") == "p13") & (pl.col("round") == 13) & (pl.col("grp") == "ctrl_late"))
print("pattern-13 label 13 bomb rows (ctrl late):", b13.height)
print(b13.group_by(["bside", "rw", "chain"]).agg(pl.len().alias("n")).sort(["bside", "rw", "chain"]))
b9n = B.filter((pl.col("pgrp") == "p13") & (pl.col("round") == 9) & (pl.col("grp") == "ctrl_late"))
print("pattern-13 matches, label 9 bomb rows (should agree with rw, no inversion at 9):", b9n.height)
print(b9n.group_by(["bside", "rw"]).agg(pl.len().alias("n")).sort(["bside", "rw"]))

# (c) joint no-plant distribution at flagged label
DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).filter(
    pl.col("player_team_code").is_in([2, 3]) & pl.col("player_id").is_not_null()).collect()
NP = (T.filter(pl.col("re_present") & pl.col("rw").is_not_null() & (pl.col("bs_planted") == 0))
      .join(MP.select(["match_key", "pgrp", "grp", "map_name"] if "map_name" in MP.columns else ["match_key", "pgrp", "grp"]),
            on="match_key", how="left").filter(pl.col("grp") == "ctrl_late"))
NP = NP.filter(pl.col("map_name") != "cs_office") if "map_name" in NP.columns else NP
NP = NP.select(K + ["rw", "pgrp", "fe_tick", "re_tick"])
D = DE.join(NP.select(K + ["fe_tick", "re_tick"]), on=K, how="inner").filter(
    (pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
ps = (D.group_by(K + ["player_team_code"]).agg(pl.col("player_id").n_unique().alias("n"))
      .pivot(on="player_team_code", index=K, values="n", aggregate_function="first")
      .rename({"2": "T_dead", "3": "CT_dead"}))
X = NP.join(ps, on=K, how="left").with_columns(pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
X = X.filter((pl.col("T_dead") + pl.col("CT_dead")) > 0)
print("\n== H3 (c): no-plant rounds (ctrl late r1 true, cs_office excluded): rw x side fully dead, by group, flagged label ==")
for lab, g in [(9, "p9"), (13, "p13"), (9, "p13"), (13, "p9"), (9, "none"), (13, "none")]:
    s = X.filter((pl.col("round") == lab) & (pl.col("pgrp") == g))
    if s.height == 0:
        print(f"label {lab} {g}: none")
        continue
    print(f"label {lab} {g}: n={s.height}")
    print(s.with_columns(
        Tall=(pl.col("T_dead") >= 5), CTall=(pl.col("CT_dead") >= 5)).group_by(["rw", "Tall", "CTall"]).agg(
        pl.len().alias("n")).sort(["rw", "Tall", "CTall"]))
