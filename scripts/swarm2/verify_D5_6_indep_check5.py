"""verify D5-6, fifth pass: is the label-13 inversion keyed to map or to recording regime?

Input: scratchpad D5_6_indep/rounds.parquet (verify_D5_6_indep_check.py) plus the header map_name.
Elimination-based inversion signal at label 13 (same definition as verify_D5_6_indep_check3.py),
split by map and regime (A = round_announce_match_start present). Regime B label 13 is the control.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_indep_check5.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_6_indep")

R = pl.read_parquet(SCR / "rounds.parquet")
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0]
H = pl.read_parquet(hf, columns=["key", "map_name"]).rename({"key": "match_key"})
R = R.join(H, on="match_key", how="left")
print("matches by regime (rounds table):", R.group_by("regime").agg(pl.col("match_key").n_unique().alias("matches")).sort("regime").to_dicts())

el = R.filter(pl.col("reason").is_in([8, 9]) & pl.col("rw").is_not_null() & pl.col("nT").is_not_null()).with_columns(
    dl=pl.when(pl.col("rw") == 2).then(pl.col("nCT")).otherwise(pl.col("nT")),
    dw=pl.when(pl.col("rw") == 2).then(pl.col("nT")).otherwise(pl.col("nCT")))
el = el.with_columns(
    kind=pl.when((pl.col("dw") >= 5) & (pl.col("dl") < 5)).then(pl.lit("inv"))
    .when((pl.col("dl") >= 5) & (pl.col("dw") < 5)).then(pl.lit("cons"))
    .otherwise(pl.lit("und")))

l13 = el.filter(pl.col("round") == 13)
print("\n== label 13 elimination inversion by map and regime (ctrl_late + aff_r13 rows, regime A and B) ==")
out = (l13.filter(pl.col("grp").is_in(["ctrl_late", "aff_r13"]))
       .group_by(["regime", "map_name"]).agg(
           pl.len().alias("elim_rounds"),
           (pl.col("kind") == "inv").sum().alias("inv"),
           (pl.col("kind") == "cons").sum().alias("cons"))
       .with_columns(decidable=pl.col("inv") + pl.col("cons"))
       .with_columns(inv_rate=(pl.col("inv") / pl.col("decidable")).round(3))
       .sort(["regime", "map_name"]))
print(out)

print("\n== label 1..12 and 14..16 elimination inversion by regime (ctrl_late + aff_r13), pooled, for contrast ==")
oth = (el.filter(pl.col("round").is_in([9, 10, 11, 12, 14, 15, 16]) & pl.col("grp").is_in(["ctrl_late", "aff_r13"]))
       .group_by("regime").agg(
           (pl.col("kind") == "inv").sum().alias("inv"),
           (pl.col("kind") == "cons").sum().alias("cons"))
       .with_columns(inv_rate=(pl.col("inv") / (pl.col("inv") + pl.col("cons"))).round(4)))
print(oth)

print("\nDONE")
