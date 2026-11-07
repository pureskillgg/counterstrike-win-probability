"""verify D5-6, sixth pass: why does last-victim disagree with round_end at label 9 in regime A (claim: 26%)?

Input: scratchpad D5_6_indep/rounds.parquet. For no-bomb rounds with deaths (claim scope), label 9,
ctrl_late and early, split by regime and by round_end reason code, with the disagreement count, and the
elimination verdict (inv / cons / und) so that a round_end inversion can be told apart from a last-victim error.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_indep_check6.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_6_indep")
R = pl.read_parquet(SCR / "rounds.parquet")

nb = R.filter((~pl.col("has_plant")) & pl.col("implied").is_not_null() & pl.col("rw").is_not_null())
nb = nb.with_columns(dis=(pl.col("implied") != pl.col("rw")))
el = nb.filter(pl.col("reason").is_in([8, 9]) & pl.col("nT").is_not_null()).with_columns(
    dl=pl.when(pl.col("rw") == 2).then(pl.col("nCT")).otherwise(pl.col("nT")),
    dw=pl.when(pl.col("rw") == 2).then(pl.col("nT")).otherwise(pl.col("nCT")))
el = el.with_columns(kind=pl.when((pl.col("dw") >= 5) & (pl.col("dl") < 5)).then(pl.lit("inv"))
                     .when((pl.col("dl") >= 5) & (pl.col("dw") < 5)).then(pl.lit("cons"))
                     .otherwise(pl.lit("und")))

print("== label 9, no-bomb rounds with deaths: last-victim disagreement by regime, grp, reason ==")
l9 = nb.filter((pl.col("round") == 9) & pl.col("grp").is_in(["ctrl_late", "early"]))
print(l9.group_by(["grp", "regime", "reason", "dis"]).agg(pl.len().alias("n")).sort(["grp", "regime", "reason", "dis"]))
print("\n== label 9 regime A ctrl_late: disagreement share among reason 8/9 rounds, split by elimination verdict ==")
print(el.filter((pl.col("round") == 9) & (pl.col("grp") == "ctrl_late") & (pl.col("regime") == "A")).group_by(
    ["kind", "dis"]).agg(pl.len().alias("n")).sort(["kind", "dis"]))

print("\n== label 9 regime A ctrl_late: no-bomb rounds, reason mix overall (all deaths, any verdict) ==")
print(nb.filter((pl.col("round") == 9) & (pl.col("grp") == "ctrl_late") & (pl.col("regime") == "A")).group_by(
    ["reason", "dis"]).agg(pl.len().alias("n")).sort(["reason", "dis"]))

print("\nDONE")
