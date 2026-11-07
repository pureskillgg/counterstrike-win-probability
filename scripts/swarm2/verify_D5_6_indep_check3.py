"""verify D5-6, third pass: per-label inversion signal from victim elimination (no last-victim ordering).

Input: scratchpad D5_6_indep/rounds.parquet written by verify_D5_6_indep_check.py.
For each round with reason 8 or 9 (elimination messages) and victim counts available:
  inv   = the side declared winner by round_end has >=5 distinct victims and the declared loser has <5
  cons  = the declared loser has >=5 distinct victims and the declared winner has <5
  und   = otherwise (both or neither side fully eliminated; time-outs, missing deaths)
Signal rate = inv / (inv + cons). A correct label gives a low rate (noise); an inverted label gives a high rate.
ASSUMPTION A2 (roster 5 per side). Regimes and groups as in the claim (A = round_announce_match_start > 0).
Also lists label-by-label rates for regime A and regime B in ctrl_late, and for aff_r13 separately.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_indep_check3.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_6_indep")
R = pl.read_parquet(SCR / "rounds.parquet")

el = R.filter(pl.col("reason").is_in([8, 9]) & pl.col("rw").is_not_null() & pl.col("nT").is_not_null()).with_columns(
    dl=pl.when(pl.col("rw") == 2).then(pl.col("nCT")).otherwise(pl.col("nT")),
    dw=pl.when(pl.col("rw") == 2).then(pl.col("nT")).otherwise(pl.col("nCT")))
el = el.with_columns(
    kind=pl.when((pl.col("dw") >= 5) & (pl.col("dl") < 5)).then(pl.lit("inv"))
    .when((pl.col("dl") >= 5) & (pl.col("dw") < 5)).then(pl.lit("cons"))
    .otherwise(pl.lit("und")))

print("== per-label elimination inversion signal, regime A and B, ctrl_late and early (labels 1..30) ==")
t = (el.filter(pl.col("round") <= 30).group_by(["grp", "regime", "round"]).agg(
    pl.len().alias("elim_rounds"),
    (pl.col("kind") == "inv").sum().alias("inv"),
    (pl.col("kind") == "cons").sum().alias("cons"),
    (pl.col("kind") == "und").sum().alias("und"))
    .with_columns(decidable=pl.col("inv") + pl.col("cons"))
    .with_columns(inv_rate=(pl.col("inv") / pl.col("decidable")).round(3))
    .sort(["grp", "regime", "round"]))
print(t.filter(pl.col("grp").is_in(["ctrl_late", "early"])))

print("\n== aff_r13 (bomb-selected) elimination inversion signal by label ==")
print(el.filter((pl.col("grp") == "aff_r13") & (pl.col("round") <= 30)).group_by("round").agg(
    pl.len().alias("elim_rounds"), (pl.col("kind") == "inv").sum().alias("inv"),
    (pl.col("kind") == "cons").sum().alias("cons")).with_columns(
    inv_rate=(pl.col("inv") / (pl.col("inv") + pl.col("cons"))).round(3)).sort("round"))

print("\n== regime A ctrl_late: labels with inv_rate > 0.5 (decidable >= 20) ==")
hi = t.filter((pl.col("grp") == "ctrl_late") & (pl.col("regime") == "A") & (pl.col("decidable") >= 20)
              & (pl.col("inv_rate") > 0.5))
print(hi.select(["round", "decidable", "inv", "cons", "inv_rate"]).sort("round"))
print("labels flagged (regime A ctrl_late):", hi["round"].to_list())

print("\nDONE")
