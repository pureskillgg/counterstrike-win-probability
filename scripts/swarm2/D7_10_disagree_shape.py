"""D7-10: shape of post-onset winner-vs-ledger disagreements (round_end winner_team_code vs min-swap ledger side).
Tests: (a) off-by-one round shift (winner equals ledger side of round r-1 or r+1); (b) per-match concentration;
(c) relation to bomb outcome presence; (d) rate by map. Pre-onset is the baseline (floor from ambiguity).
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
X = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger_dp.parquet"))
L = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet")).select(["match_key", "round", "ledger_side"])
Lm = L.rename({"round": "rr", "ledger_side": "ls_rr"})
# neighbour ledger sides
prev = L.with_columns((pl.col("round") + 1).alias("round")).rename({"ledger_side": "ls_prev"})  # ledger of round r-1 at row r
nxt = L.with_columns((pl.col("round") - 1).alias("round")).rename({"ledger_side": "ls_next"})  # ledger of round r+1 at row r
Y = X.join(prev.select(["match_key", "round", "ls_prev"]), on=["match_key", "round"], how="left") \
     .join(nxt.select(["match_key", "round", "ls_next"]), on=["match_key", "round"], how="left")
Y = Y.filter(pl.col("w_ok").is_not_null())
print("rows with unique ledger side and winner code:", Y.height)
print("\n(a) post-onset disagreements: does winner equal ledger of r-1 / r+1?")
P = Y.filter((pl.col("period") == "post") & (~pl.col("w_ok")))
print("post disagreements:", P.height,
      "| winner == ledger(r-1):", P.filter(pl.col("winner_team_code") == pl.col("ls_prev")).height,
      "| winner == ledger(r+1):", P.filter(pl.col("winner_team_code") == pl.col("ls_next")).height)

print("\n(b) per-match concentration (post period, rows with unique ledger):")
pm = (Y.filter(pl.col("period") == "post").group_by("match_key")
        .agg(pl.len().alias("n"), (~pl.col("w_ok")).sum().alias("dis")))
print("matches with >=1 disagreement:", pm.filter(pl.col("dis") > 0).height, "of", pm.height)
print("disagreement count per match distribution (dis: #matches):", pm.filter(pl.col("dis") > 0).group_by("dis").len().sort("dis").head(15).rows())
print("fraction of rounds disagreeing among matches with >=1:",
      round(pm.filter(pl.col("dis") > 0)["dis"].sum() / pm.filter(pl.col("dis") > 0)["n"].sum(), 3))
print("matches where ALL rounds disagree (n>=5):", pm.filter((pl.col("dis") == pl.col("n")) & (pl.col("n") >= 5)).height)

print("\n(c) post disagreements by bomb outcome present:")
print(Y.filter(pl.col("period") == "post").group_by([pl.col("bomb_side").is_not_null().alias("bomb"), "w_ok"]).len().sort(["bomb", "w_ok"]))

print("\n(d) post disagreement rate by map (unique-ledger rounds):")
print(Y.filter(pl.col("period") == "post").group_by("map_name").agg(pl.len().alias("n"), (~pl.col("w_ok")).sum().alias("dis"),
      (~pl.col("w_ok")).mean().round(4).alias("rate")).sort("rate", descending=True).rows())
print("pre rate by map:")
print(Y.filter(pl.col("period") == "pre").group_by("map_name").agg(pl.len().alias("n"), (~pl.col("w_ok")).sum().alias("dis"),
      (~pl.col("w_ok")).mean().round(4).alias("rate")).sort("rate", descending=True).rows())

print("\n(e) post disagreement rate by round number (top 10 rounds by count):")
print(Y.filter(pl.col("period") == "post").group_by("round").agg(pl.len().alias("n"), (~pl.col("w_ok")).sum().alias("dis")).sort("dis", descending=True).head(10).rows())

print("\n(f) per-day post disagreement rate (match_date UTC):")
print(Y.filter(pl.col("period") == "post").with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("d")).group_by("d")
      .agg(pl.len().alias("n"), (~pl.col("w_ok")).sum().alias("dis"), (~pl.col("w_ok")).mean().round(3).alias("rate")).sort("d").rows())
