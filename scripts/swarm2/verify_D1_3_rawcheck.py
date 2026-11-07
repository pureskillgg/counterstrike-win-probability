"""verify D1-3: compare round_end winner_team_code in the tome vs the raw E: source for ONE affected match (read-only).

Match: csds/2024/02/12/e025c2e1-2251-4c9a-90be-a68cab42d04d (tome winner disagrees with the score side at round 13).
Raw: E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12\<match_id>\round_end*  (read only).
Prints the per-round winner_team_code for both, and the round_state score-side at 13 from raw.
"""
import glob
import sys
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
MID = "e025c2e1-2251-4c9a-90be-a68cab42d04d"
RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12") / MID
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds\basic_round_end.2023-12-28,2024-02-25")
print("raw dir exists:", RAW.exists())
rawfiles = glob.glob(str(RAW / "**" / "round_end*"), recursive=True)
print("raw round_end files:", rawfiles[:5])
tomefiles = sorted(glob.glob(str(TOME / "dataframe_*")))
tome = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code"]) for f in tomefiles], how="vertical_relaxed")
tome = tome.filter(pl.col("match_key").str.contains(MID)).sort("round")
print("tome rows for match:", tome.height)
if rawfiles:
    raw = pl.read_parquet(rawfiles[0])
    print("raw columns:", raw.columns)
    cols = [c for c in ["round", "winner_team_code", "win_reason_message", "player_count"] if c in raw.columns]
    rr = raw.select(cols).sort("round")
    j = tome.select(["round", pl.col("winner_team_code").alias("tome_winner")]).join(
        rr.select(["round", pl.col("winner_team_code").alias("raw_winner")]), on="round", how="full", coalesce=True).sort("round")
    print(j)
    print("tome == raw winner in", int((j["tome_winner"] == j["raw_winner"]).sum()), "of", j.height, "rounds")
