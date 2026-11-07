"""verify D7-2 (04): the second FACEIT match (m_56c0f3b3, build 9961, 2024-02-12 03:57:08 UTC) is the natural test of build vs server.
Check whether it enters the first-after-swap set and what its ledger / winner rows show. ASSUMPTION: same definitions as 01.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
LOG = ROOT / "logs" / "swarm2"
X = pl.read_parquet(str(LOG / "D7_rounds_with_ledger_dp.parquet"))
L = pl.read_parquet(str(LOG / "D7_ledger_dp.parquet"))
keys = X.filter(pl.col("match_key").str.contains("m_56c0f3b3") | pl.col("match_key").str.contains("m_98cefeb6"))["match_key"].unique().to_list()
print("matched keys:", [k[:50] for k in keys])
for k in keys:
    print("\n==", k[:50])
    rows = X.filter(pl.col("match_key") == k).select(["round", "winner_team_code", "ledger_side", "w_ok", "bomb_side", "status", "mdt"]).sort("round")
    print(rows)
    lg = L.filter(pl.col("match_key") == k).sort("round")
    print("ledger rows:", lg.height, "| swaps at rounds:", lg.filter(pl.col("swap") == True)["round"].to_list(), "| status:", lg["status"].unique().to_list())
print("\nfull m_56c0f3b3 ledger rows (any round):")
print(L.filter(pl.col("match_key").str.contains("m_56c0f3b3")).sort("round"))
print("total X rows for m_56c0f3b3:", X.filter(pl.col("match_key").str.contains("m_56c0f3b3")).height)
