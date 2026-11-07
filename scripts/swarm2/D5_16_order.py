"""D5 step 16: systematic ordering check on round_state freeze-end and round_end ticks, by group, labels 11..15.

For consecutive labels r and r+1 with freeze-end (fe) and round_end (end) ticks present:
  A  fe(r) < end(r)            (live window starts before the round ends)
  B  end(r) < fe(r+1)          (round r ends before round r+1 starts living)
  C  end(r) < end(r+1)         (round_end ticks are increasing)
Counts per group, all labels and labels 11..15 only.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_16_order.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet").select(["match_key", "round", "fe_tick", "end_tick", "in_fe", "in_end"]).sort(
    ["match_key", "round"])
M = pl.read_parquet(SCR / "matches.parquet").select(["match_key", "aff_r13", "day"])
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.with_columns(grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
N = R.with_columns(nfe=pl.col("fe_tick").shift(-1).over("match_key"),
                   nend=pl.col("end_tick").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"))
N = N.filter((pl.col("nr") == pl.col("round") + 1)).join(M.select(["match_key", "grp"]), on="match_key", how="left")
N = N.with_columns(
    A=pl.when(pl.col("fe_tick").is_not_null() & pl.col("end_tick").is_not_null())
    .then(pl.col("fe_tick") < pl.col("end_tick")).otherwise(None),
    B=pl.when(pl.col("end_tick").is_not_null() & pl.col("nfe").is_not_null())
    .then(pl.col("end_tick") < pl.col("nfe")).otherwise(None),
    C=pl.when(pl.col("end_tick").is_not_null() & pl.col("nend").is_not_null())
    .then(pl.col("end_tick") < pl.col("nend")).otherwise(None))
for name, sub in [("all labels", N), ("labels 11..15", N.filter(pl.col("round").is_between(11, 15)))]:
    print(f"== {name} ==")
    print(sub.group_by("grp").agg(
        pl.len().alias("pairs"),
        pl.col("A").is_not_null().sum().alias("A_testable"), (pl.col("A") == False).sum().alias("A_violations"),  # noqa: E712
        pl.col("B").is_not_null().sum().alias("B_testable"), (pl.col("B") == False).sum().alias("B_violations"),  # noqa: E712
        pl.col("C").is_not_null().sum().alias("C_testable"), (pl.col("C") == False).sum().alias("C_violations"),  # noqa: E712
    ).sort("grp"))
print("\n== affected matches, labels 11..15, detail (pairs (r, r+1)) ==")
print(N.filter((pl.col("grp") == "aff_r13") & pl.col("round").is_between(11, 15)).group_by("round").agg(
    pl.len().alias("pairs"), pl.col("A").sum().alias("A_true"), pl.col("B").sum().alias("B_true"),
    pl.col("C").sum().alias("C_true")).sort("round"))
