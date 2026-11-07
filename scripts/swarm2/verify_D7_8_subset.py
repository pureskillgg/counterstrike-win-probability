"""verify_D7_8 part 6: reconcile the claim's pattern counts (D7_9 uses D7 rounds with player_count==20) with the full match set (stdout only)."""
import io
import contextlib
import sys
from pathlib import Path

import polars as pl

src = Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_8_main.py").read_text(encoding="utf-8")
src = src.replace('sys.stdout.reconfigure(encoding="utf-8")', "pass")
g = {"__name__": "verify_main_subset"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, "verify_D7_8_main.py", "exec"), g)
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(260)
PM, PI, ROOT = g["PM"], g["PI"], g["ROOT"]
X7 = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger_dp.parquet"), columns=["match_key", "round", "player_count"])
keys_D7_9 = set(X7.filter(pl.col("player_count") == 20)["match_key"].unique().to_list())
print("matches in D7_9 pattern universe (rounds with player_count==20):", len(keys_D7_9))
print("matches in D7 rounds file at all:", X7["match_key"].n_unique())
RS = PI.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n_players"))
S = PM.join(RS, on="match_key", how="left").with_columns(pl.col("match_key").is_in(list(keys_D7_9)).alias("in_D7_9"))
print("\nDP pattern counts, in D7_9 universe (should be claim's 3111/734/253/97/21 if matched):")
print(S.filter(pl.col("in_D7_9")).group_by("dp_pat").len().sort("len", descending=True).rows())
print("DP pattern counts, NOT in D7_9 universe:", S.filter(~pl.col("in_D7_9")).group_by(["dp_pat", "n_players"]).len().sort("len", descending=True).head(8).rows())
print("\nplayer-based pattern counts, same D7_9 universe:", S.filter(pl.col("in_D7_9")).group_by("pl_pat").len().sort("len", descending=True).rows())
T = S.filter(pl.col("in_D7_9") & (pl.col("n_players") == 10))
print("\n10-player matches in D7_9 universe:", T.height)
print("  DP pattern counts:", T.group_by("dp_pat").len().sort("len", descending=True).rows())
print("  player pattern counts:", T.group_by("pl_pat").len().sort("len", descending=True).rows())
print("  exact agreement (pl == dp):", T.filter(pl.col("pl_pat") == pl.col("dp_pat")).height)
print("  crosstab pl x dp:", T.group_by(["pl_pat", "dp_pat"]).len().sort("len", descending=True).rows())
