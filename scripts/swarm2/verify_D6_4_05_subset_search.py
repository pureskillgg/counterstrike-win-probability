"""D6-4 verifier, step 5: search plausible 'unambiguous score' definitions for the claim's 370 and 274.
Only reports which definition (if any) reproduces 370 affected and 274 control. Nothing is written.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet"))
r13 = j.filter((pl.col("player_count") == 20) & (pl.col("round") == 13) & pl.col("bomb_side").is_not_null()
               & pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)]))
r13 = r13.with_columns(
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("aff"),
    (pl.col("t_post") - pl.col("ct_post")).abs().alias("gap_post") if "t_post" in r13.columns else pl.lit(None).alias("gap_post"))
# pre-round gap from freeze_end(13) is not in the join; use post gap only
cands = {
    "score_known": pl.col("score_side").is_not_null(),
    "unambig(dT+dCT==1)": pl.col("score_unambig") == True,
    "post_gap>=2": pl.col("gap_post") >= 2,
    "post_gap>=2 & known": (pl.col("gap_post") >= 2) & pl.col("score_side").is_not_null(),
    "dT,dCT both not null": pl.col("dT").is_not_null() & pl.col("dCT").is_not_null(),
}
for name, expr in cands.items():
    sub = r13.filter(expr)
    a = sub.filter(pl.col("aff"))
    c = sub.filter(~pl.col("aff"))
    print(f"{name:24s} affected={a.height:4d} control={c.height:4d}  affected score==bomb={(a['score_side']==a['bomb_side']).sum() if a.height else 0}")
print("target: affected 370, control 274")
print("post-round gap distribution (affected):", r13.filter("aff" and pl.col("aff")).group_by("gap_post").len().sort("gap_post").head(10).to_dicts())
