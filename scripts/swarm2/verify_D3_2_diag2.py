"""D3-2 verifier diagnostic 2: check the round-14 'RE same as next' anomaly in verify_D3_2_diag.py (a).

For days 12-25, for each match and pair (r, r+1) with r in {12,13,14,15}, cross-tabulate
  score(r) == RE(r)   and   score(r+1) == RE(r+1)
using only rows where all four values are defined, and report RE(r)==RE(r+1) against score(r)==score(r+1).
Read-only on the G: 2024 tome.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def load(ch, cols):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


RE = load("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RS = load("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
DAY = pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64)

fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"])
      .agg(pl.col("t_score").first().alias("t_b"), pl.col("ct_score").first().alias("c_b")))
nxt = fe.select(["match_key", (pl.col("round") - 1).alias("round"),
                 pl.col("t_b").alias("t_a"), pl.col("c_b").alias("c_a")])
sc = fe.join(nxt, on=["match_key", "round"], how="left").with_columns(
    (pl.col("t_a") - pl.col("t_b")).alias("dt"), (pl.col("c_a") - pl.col("c_b")).alias("dc")).with_columns(
    pl.when((pl.col("dc") == 1) & (pl.col("dt") == 0)).then(pl.lit(CT, dtype=pl.Int64))
    .when((pl.col("dt") == 1) & (pl.col("dc") == 0)).then(pl.lit(T, dtype=pl.Int64))
    .otherwise(pl.lit(None, dtype=pl.Int64)).alias("sc_win"))

# one RE row per (match, round): the pc==20 row with the smallest tick (as in verify_D3_2_main)
re20 = (RE.filter(pl.col("player_count") == 20).sort("tick").group_by(["match_key", "round"])
        .agg(pl.col("winner_team_code").first().alias("re_w")))

W = (re20.join(sc.select(["match_key", "round", "sc_win"]), on=["match_key", "round"], how="left")
     .with_columns(DAY.alias("day")))

print("rows per round, days 12-25: RE defined, score defined")
print(W.filter(pl.col("day") >= 12).group_by("round").agg(
    pl.len().alias("n_re"), pl.col("sc_win").is_not_null().sum().alias("n_sc"),
    (pl.col("sc_win") == pl.col("re_w")).sum().alias("agree_when_both")).sort("round").head(20))

for r in [12, 13, 14, 15]:
    A = W.filter(pl.col("round") == r).select(["match_key", "day", pl.col("re_w").alias("re_r"), pl.col("sc_win").alias("sc_r")])
    Bn = W.filter(pl.col("round") == r + 1).select(["match_key", pl.col("re_w").alias("re_r1"), pl.col("sc_win").alias("sc_r1")])
    J = A.join(Bn, on="match_key", how="inner").filter(pl.col("day") >= 12)
    K = J.filter(pl.col("sc_r").is_not_null() & pl.col("sc_r1").is_not_null())
    print(f"\n== pair ({r},{r+1}), days 12-25, rows with score defined for both: n={K.height}")
    print("  RE(r)==RE(r+1):", int((K["re_r"] == K["re_r1"]).sum()),
          "  score(r)==score(r+1):", int((K["sc_r"] == K["sc_r1"]).sum()))
    print("  score(r)==RE(r):", int((K["sc_r"] == K["re_r"]).sum()), " score(r+1)==RE(r+1):",
          int((K["sc_r1"] == K["re_r1"]).sum()))
    print(K.group_by([(K["sc_r"] == K["re_r"]).alias("r_ok"), (K["sc_r1"] == K["re_r1"]).alias("r1_ok")])
          .len().sort(["r_ok", "r1_ok"]))

print("\n== DONE")
