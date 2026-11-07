"""verify_D7_5_100_r27: why the within-round and DP labels disagree with bomb and death checks at round 27.

For each round-27 row (all player counts): period, whether the DP sees a side swap between 27 and 28 (fas28),
whether round_officially_ended(27) exists (within available), and agreement of code / within / DP with the
bomb outcome and with the last-death winner (death check as in verify_D7_5_60_death_winner.py).
"""
import sys
import glob
from pathlib import Path
from collections import Counter
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet")).sort(["match_key", "round"])
R = R.with_columns(
    pl.col("orient").shift(-1).over("match_key").alias("orient_next"),
    pl.col("round").shift(-1).over("match_key").alias("round_next"),
)
fs = sorted(glob.glob(str(BASE / f"basic_player_death{SUF}" / "dataframe_*")))
D = pl.concat([pl.scan_parquet(f) for f in fs]).select(["match_key", "round", "tick", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([2, 3])).collect()
last = D.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("lt"))
lastd = D.join(last, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("lt"))
ties = lastd.group_by(["match_key", "round"]).agg(pl.col("player_team_code").n_unique().alias("nu"))
lastd = lastd.group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("vs")).join(
    ties.filter(pl.col("nu") == 1).select(["match_key", "round"]), on=["match_key", "round"])
lastd = lastd.with_columns(pl.when(pl.col("vs") == 2).then(3).otherwise(2).alias("W_death"))
X = R.filter(pl.col("round") == 27).join(lastd.select(["match_key", "round", "W_death"]), on=["match_key", "round"], how="left")
X = X.with_columns(
    ((pl.col("orient_next") != pl.col("orient")) & (pl.col("round_next") == 28)).alias("fas28"),
    (pl.col("W_within").is_not_null()).alias("has_within"),
    (pl.col("W_dp").is_not_null()).alias("has_dp"),
)
print("round-27 rows:", X.height)
for p in ["pre", "post"]:
    s = X.filter(pl.col("period") == p)
    print(f"\n== {p} ==")
    for fas in [True, False]:
        for hw in [True, False]:
            t = s.filter((pl.col("fas28") == fas) & (pl.col("has_within") == hw))
            if t.height == 0:
                continue
            nb = t.filter(pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0))
            nd = t.filter(pl.col("W_death").is_not_null())
            print(f"fas28={fas!s:5s} within={hw!s:5s} n={t.height:4d} | "
                  f"code==within {t.filter(pl.col('W_code') == pl.col('W_within')).height}/{t.filter(pl.col('has_within')).height} | "
                  f"dp==within {t.filter(pl.col('W_dp') == pl.col('W_within')).height}/{t.filter(pl.col('has_within') & pl.col('has_dp')).height} | "
                  f"bomb rows {nb.height}: code {nb.filter(pl.col('W_code') == pl.col('W_bomb')).height}, "
                  f"within {nb.filter(pl.col('W_within') == pl.col('W_bomb')).height}, dp {nb.filter(pl.col('W_dp') == pl.col('W_bomb')).height} | "
                  f"death rows {nd.height}: code {nd.filter(pl.col('W_code') == pl.col('W_death')).height}, "
                  f"within {nd.filter(pl.col('W_within') == pl.col('W_death')).height}, dp {nd.filter(pl.col('W_dp') == pl.col('W_death')).height}")
