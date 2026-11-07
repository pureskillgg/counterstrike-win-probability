# X15 diagnostics: sign sanity, round-level equipment signal, static-feature distinctness,
# and hero deaths split by whether the victim side lost its last rifle/AWP holder.
import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

SCR = "C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad/"
g = pl.read_parquet(SCR + "X15_grid.parquet")
dd = pl.read_parquet(SCR + "X15_hero_dd.parquet")

print("=== sign sanity: mean signed drop by victim team (positive = worse for victim side) ===")
print(dd.group_by("team_code").agg(pl.col("drop_a").mean().alias("drop_a"), pl.col("drop_c").mean().alias("drop_c"), pl.len().alias("n")).sort("team_code"))

print("\n=== round-level freeze-end equipment signal (one row per round) ===")
rd = g.group_by(["match_key", "round"]).agg(pl.col("ct_eq_fe").first(), pl.col("t_eq_fe").first(), pl.col("y").first())
rd = rd.with_columns((pl.col("ct_eq_fe") - pl.col("t_eq_fe")).alias("eq_diff"))
print("rounds", rd.height, " mean ct_eq_fe", round(rd["ct_eq_fe"].mean(), 1), " mean t_eq_fe", round(rd["t_eq_fe"].mean(), 1))
print("distinct (ct_eq_fe, t_eq_fe) pairs:", rd.select(["ct_eq_fe", "t_eq_fe"]).n_unique(), "of", rd.height, "rounds")
print("corr(eq_diff, ct_win) across rounds:", round(float(np.corrcoef(rd["eq_diff"].to_numpy(), rd["y"].to_numpy())[0, 1]), 4))
print("AUC of eq_diff alone for ct_win:", round(roc_auc_score(rd["y"].to_numpy(), rd["eq_diff"].to_numpy()), 4))
print("rows per round (mean):", round(g.height / rd.height, 1))

print("\n=== hero deaths split by side rifle/AWP holders after death (side_ra_t1 == 0 means victim side has no rifle/AWP left) ===")
dd = dd.with_columns((pl.col("side_ra_t1") == 0).alias("side_no_rifle_after"))
tab = (dd.group_by(["victim_class", "side_no_rifle_after"])
       .agg(pl.len().alias("n"), pl.col("drop_a").mean().alias("drop_a"), pl.col("drop_c").mean().alias("drop_c"))
       .sort(["victim_class", "side_no_rifle_after"]))
print(tab)

print("\n=== hero deaths split by side rifle/AWP holders before death (side_ra_t) ===")
tab2 = (dd.filter(pl.col("victim_class").is_in(["AWP", "rifle"]))
        .group_by(["victim_class", "side_ra_t"])
        .agg(pl.len().alias("n"), pl.col("drop_a").mean().alias("drop_a"), pl.col("drop_c").mean().alias("drop_c"))
        .sort(["victim_class", "side_ra_t"]))
print(tab2)

print("\n=== drops by victim side rifle/AWP count before death, all victims (model c) ===")
tab3 = (dd.group_by("side_ra_t")
        .agg(pl.len().alias("n"), pl.col("drop_a").mean().alias("drop_a"), pl.col("drop_c").mean().alias("drop_c"))
        .sort("side_ra_t"))
print(tab3)
