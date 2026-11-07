"""Debug: regime B round-1 and round-2 rows for a few differing matches (read only, scratchpad caches)."""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10")
T, CT = 2, 3
FE = pl.read_parquet(SCR / "fe.parquet")
RE = pl.read_parquet(SCR / "re.parquet")
MR = pl.read_parquet(SCR / "mr.parquet")
print("FE round dtype:", FE.schema["round"], "; RE round dtype:", RE.schema["round"])
print("RE rows regime B by round label (first 4):", RE.join(MR.select(["match_key", "regime"]), on="match_key").filter(pl.col("regime") == "B").group_by("round").len().sort("round").head(4).rows())
print("FE rows regime B by round label (first 4):", FE.join(MR.select(["match_key", "regime"]), on="match_key").filter(pl.col("regime") == "B").group_by("round").len().sort("round").head(4).rows())
bad = ["csds/2024/02/"]  # placeholder
X = FE.join(RE, on=["match_key", "round"], how="left").join(MR.select(["match_key", "regime"]), on="match_key").filter(pl.col("regime") == "B")
print("B freeze-end rows:", X.height, "; with round_end winner:", X.filter(pl.col("winner_team_code").is_not_null()).height)
print("B rounds where freeze-end exists but no round_end (by label, first 6):", X.filter(pl.col("winner_team_code").is_null()).group_by("round").len().sort("round").head(6).rows())
