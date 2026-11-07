"""D5 step 18: do matches with a side-field swap at 8->9 show any pause/restart-type event signature?

Compares, for labels 7..10, the per-match counts of round_state event types between:
  swap8  = matches with a side_swap at 8->9 (D5_14 rule, round_end winner 8)
  other  = all other matches (same regime)
Event types checked: round_freeze_end, round_prestart, round_officially_ended, cs_pre_restart,
round_announce_* (all), begin_new_match, cs_win_panel_round, buytime_ended.
Also: round_end rows and round_start rows per match around the swap (extra or missing rows).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_18_swap8_events.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet").sort(["match_key", "round"])
M = pl.read_parquet(SCR / "matches.parquet")
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
mk = R.group_by("match_key").agg(pl.col("n_round_announce_match_start").fill_null(0).sum().alias("mstart"))
M = (M.join(mk, on="match_key", how="left")
     .with_columns(regime=pl.when(pl.col("mstart") > 0).then(pl.lit("A")).otherwise(pl.lit("B")),
                   grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other"))))

# swap8 flag (same as D5_17)
N8 = R.filter(pl.col("round") == 8).select(["match_key", "fe_t_score", "fe_ct_score", "winner"])
nx = R.filter(pl.col("round") == 9).select(["match_key", pl.col("fe_t_score").alias("t9"), pl.col("fe_ct_score").alias("ct9")])
J8 = N8.join(nx, on="match_key", how="inner").filter(pl.col("winner").is_not_null() & pl.col("t9").is_not_null())
a = pl.when(pl.col("winner") == 2).then(1).otherwise(0)
b = pl.when(pl.col("winner") == 3).then(1).otherwise(0)
J8 = J8.with_columns(ns=(pl.col("t9") == pl.col("fe_t_score") + a) & (pl.col("ct9") == pl.col("fe_ct_score") + b),
                     sw=(pl.col("t9") == pl.col("fe_ct_score") + b) & (pl.col("ct9") == pl.col("fe_t_score") + a))
sw8 = J8.filter(pl.col("sw") & ~pl.col("ns")).select("match_key").with_columns(swap8=pl.lit(True))
M = M.join(sw8, on="match_key", how="left").with_columns(swap8=pl.col("swap8").fill_null(False))

ev_cols = [c for c in R.columns if c.startswith("n_") and c not in ("n_start_rows", "n_fe_rows", "n_end_rows",
                                                                  "n_start", "n_fe", "n_distinct_winner")]
ev_cols = [c for c in ev_cols if c.startswith("n_round_") or c.startswith("n_cs_") or c in ("n_begin_new_match", "n_buytime_ended")]
print("event columns checked:", ev_cols)
W = R.filter(pl.col("round").is_between(7, 10)).group_by("match_key").agg(
    [pl.col(c).fill_null(0).sum().alias(c) for c in ev_cols]
    + [pl.col("in_end").sum().alias("end_rows_7_10"), pl.col("in_start").sum().alias("start_rows_7_10"),
       pl.col("in_fe").sum().alias("fe_rows_7_10")])
X = M.select(["match_key", "grp", "regime", "swap8"]).join(W, on="match_key", how="left")
for reg in ["A", "B"]:
    sub = X.filter(pl.col("regime") == reg)
    print(f"\n== regime {reg}: mean per-match event counts over labels 7..10, swap8 vs other ==")
    print(sub.group_by("swap8").agg(
        pl.len().alias("matches"),
        *[pl.col(c).mean().round(2).alias(c) for c in ev_cols],
        pl.col("end_rows_7_10").mean().round(2).alias("end_rows"),
        pl.col("start_rows_7_10").mean().round(2).alias("start_rows"),
        pl.col("fe_rows_7_10").mean().round(2).alias("fe_rows")).sort("swap8").transpose(include_header=True))
print("\nregime A swap8 matches: matches with any event-count anomaly (fe rows != 4, end rows != 4 in labels 7..10):")
print(X.filter((pl.col("regime") == "A") & pl.col("swap8")).select(
    (pl.col("fe_rows_7_10") != 4).sum().alias("fe_anom"), (pl.col("end_rows_7_10") != 4).sum().alias("end_anom"),
    (pl.col("start_rows_7_10") != 4).sum().alias("start_anom"), pl.len().alias("n")))
print("regime A other matches, same check:")
print(X.filter((pl.col("regime") == "A") & ~pl.col("swap8")).select(
    (pl.col("fe_rows_7_10") != 4).sum().alias("fe_anom"), (pl.col("end_rows_7_10") != 4).sum().alias("end_anom"),
    (pl.col("start_rows_7_10") != 4).sum().alias("start_anom"), pl.len().alias("n")))
