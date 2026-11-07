"""D6-5 verifier, part 6: raw E: round_end for one affected match vs the tome copy (read-only, one match)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
MID = "40f9484a-d796-42b2-b32e-dead7d12bc39"
RAW_FILE = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\19") / MID / "round_end"
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
try:
    rre = pl.read_parquet(RAW_FILE)
    print("raw round_end parquet columns:", rre.columns, "rows:", rre.height)
    cols = [c for c in ["round", "winner_team_code", "win_reason_message", "tick"] if c in rre.columns]
    print(rre.select(cols).sort("round").head(20))
except Exception as e:
    print("raw read failed:", type(e).__name__, str(e)[:300])
dirs = sorted(TOME.glob("basic_round_end.*"))
tre = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(dirs[0] / "dataframe_*")))], how="vertical_relaxed")
tre = tre.filter(pl.col("match_key").str.contains(MID)).select(["round", "winner_team_code", "win_reason_message", "tick"]).sort("round")
print("tome round_end for this match:")
print(tre.head(20))
