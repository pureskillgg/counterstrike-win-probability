"""verify_D7_8 part 4: team sizes per round for 4-player matches, max rounds by swap class, 2026 header max_rounds check (stdout only)."""
import io
import contextlib
import glob
import sys
from pathlib import Path

import polars as pl

src = Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_8_main.py").read_text(encoding="utf-8")
src = src.replace('sys.stdout.reconfigure(encoding="utf-8")', "pass")
g = {"__name__": "verify_main_extra"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, "verify_D7_8_main.py", "exec"), g)
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(260)
PI, H, PM, FE, SA = (g[k] for k in ["PI", "H", "PM", "FE", "SA"])
RS = PI.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n_players"))

print("== E1. players per side per round, by roster size (team_code 2 = T, 3 = CT)")
side_counts = PI.group_by(["match_key", "round", "team_code"]).len().with_columns(pl.col("len").alias("n"))
per = (side_counts.join(RS, on="match_key", how="left")
       .group_by(["n_players", "team_code"]).agg(pl.col("n").median().alias("median_per_side"), pl.col("n").min().alias("min"), pl.col("n").max().alias("max"))
       .sort(["n_players", "team_code"]))
print(per.rows())

print("\n== E2. max round by roster size and by player-based swap class")
MR = FE.group_by("match_key").agg(pl.col("round").max().alias("max_round"))
M2 = PM.join(RS, on="match_key", how="left").join(MR, on="match_key", how="left")
print("max_round by (pl_pat, n_players) top:", M2.group_by(["pl_pat", "n_players"]).agg(pl.col("max_round").min().alias("min_max"), pl.col("max_round").max().alias("max_max"), pl.len().alias("n")).sort("n", descending=True).head(10).rows())
print("pl '' (no swap) matches: max_round distribution:", M2.filter(pl.col("pl_pat") == "").group_by("max_round").len().sort("max_round").rows())
print("max_round overall for 4-player matches:", M2.filter(pl.col("n_players") == 4).group_by("max_round").len().sort("max_round").rows())

print("\n== E3. 2026 header: does it carry max_rounds? (read-only, first file)")
for tp in ["tomes/2026-10-01_to_08/tome/csds/header"]:
    fs = sorted(glob.glob(str(g["ROOT"] / tp / "dataframe_*")))
    if fs:
        import polars as _p
        sch = _p.read_parquet_schema(fs[0])
        print(tp, "| files:", len(fs), "| has max_rounds:", "max_rounds" in sch, "| columns:", list(sch.keys())[:40])
    else:
        print(tp, "NOT FOUND")
