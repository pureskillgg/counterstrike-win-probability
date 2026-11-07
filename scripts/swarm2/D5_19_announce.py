"""D5 step 19: in-game announcements (last round of half, match point) for the 8->9 swap matches.

Shows, for each announcement type, the labels where it occurs, split by swap8 (side_swap at 8->9) and regime.
Also: final round label and round_end row count for swap8 matches (are they 16-round / 8-round halves?).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_19_announce.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
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
N8 = R.filter(pl.col("round") == 8).select(["match_key", "fe_t_score", "fe_ct_score", "winner"])
nx = R.filter(pl.col("round") == 9).select(["match_key", pl.col("fe_t_score").alias("t9"), pl.col("fe_ct_score").alias("ct9")])
J8 = N8.join(nx, on="match_key", how="inner").filter(pl.col("winner").is_not_null() & pl.col("t9").is_not_null())
a = pl.when(pl.col("winner") == 2).then(1).otherwise(0)
b = pl.when(pl.col("winner") == 3).then(1).otherwise(0)
J8 = J8.with_columns(ns=(pl.col("t9") == pl.col("fe_t_score") + a) & (pl.col("ct9") == pl.col("fe_ct_score") + b),
                     sw=(pl.col("t9") == pl.col("fe_ct_score") + b) & (pl.col("ct9") == pl.col("fe_t_score") + a))
sw8 = J8.filter(pl.col("sw") & ~pl.col("ns")).select("match_key").with_columns(swap8=pl.lit(True))
M = M.join(sw8, on="match_key", how="left").with_columns(swap8=pl.col("swap8").fill_null(False))

ann = ["n_round_announce_last_round_half", "n_round_announce_match_point", "n_round_announce_final",
       "n_round_announce_match_start", "n_cs_win_panel_match"]
X = R.join(M.select(["match_key", "regime", "swap8", "grp"]), on="match_key", how="left")
ann_rows = (X.filter(pl.col("n_round_announce_last_round_half").fill_null(0) + pl.col("n_round_announce_match_point").fill_null(0) > 0)
            .group_by(["regime", "swap8", "round"]).agg(
                pl.col("n_round_announce_last_round_half").fill_null(0).sum().alias("last_round_half"),
                pl.col("n_round_announce_match_point").fill_null(0).sum().alias("match_point"),
                pl.len().alias("rows")).sort(["regime", "swap8", "round"]))
print("announcement counts by round label (only labels with any announcement):")
print(ann_rows)

print("\nper-match counts of last_round_half and match_point by swap8 and regime:")
per = X.group_by(["match_key"]).agg(
    pl.col("n_round_announce_last_round_half").fill_null(0).sum().alias("lrh"),
    pl.col("n_round_announce_match_point").fill_null(0).sum().alias("mp"),
    pl.col("round").max().alias("max_label")).join(M.select(["match_key", "regime", "swap8", "grp"]), on="match_key", how="left")
print(per.group_by(["regime", "swap8"]).agg(pl.len().alias("matches"), (pl.col("lrh") > 0).sum().alias("with_lrh"),
                                             (pl.col("mp") > 0).sum().alias("with_mp"),
                                             pl.col("max_label").mean().round(2).alias("mean_max_label")).sort(["regime", "swap8"]))
print("\nmatches with last_round_half announced at labels 7..10 (swap8 vs other):")
print(X.filter(pl.col("round").is_between(7, 10) & (pl.col("n_round_announce_last_round_half").fill_null(0) > 0))
      .group_by(["regime", "swap8", "round"]).agg(pl.len().alias("matches")).sort(["regime", "swap8", "round"]))
