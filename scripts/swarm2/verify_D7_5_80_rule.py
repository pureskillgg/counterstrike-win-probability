"""verify_D7_5_80_rule: exact rule for which post-change rounds are inverted in winner_team_code.

Groups code != within by round (9, 13, 25, 27, 28), by the match's exact last round (maxr), and by whether the
min-swap ledger has a side swap between round r-1 and r (fas flag). Also reports the matched bomb rows.
"""
import sys
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet")).sort(["match_key", "round"])
R = R.with_columns(
    pl.col("orient").shift(1).over("match_key").alias("orient_prev"),
    pl.col("round").shift(1).over("match_key").alias("round_prev"),
)
R = R.with_columns(
    ((pl.col("orient") != pl.col("orient_prev")) & (pl.col("round_prev") == pl.col("round") - 1)).alias("fas"),
    (pl.col("W_code") != pl.col("W_within")).alias("dis"),
)
both = R.filter(pl.col("W_code").is_not_null() & pl.col("W_within").is_not_null())
post = both.filter(pl.col("period") == "post")
print("== post, rounds 9 and 13: inverted (code != within) by fas flag and maxr ==")
for rnd in [9, 13]:
    s = post.filter(pl.col("round") == rnd)
    g = s.group_by(["fas", "maxr"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort(["fas", "maxr"])
    print(f"round {rnd}: (fas, maxr, n, inverted)")
    print(g.rows())
print("\n== post, rounds 25, 27, 28: inverted by maxr ==")
for rnd in [25, 27, 28]:
    s = post.filter(pl.col("round") == rnd)
    print(f"round {rnd}: (maxr, n, inverted)", s.group_by("maxr").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("maxr").rows())
print("\n== all periods: fas flag by round (player counts all) ==")
print(R.filter(pl.col("fas")).group_by(["period", "round"]).len().sort(["period", "round"]).rows())
print("\n== post: any round, code != within, count by (round, fas) ==")
print(post.filter(pl.col("dis")).group_by(["round", "fas"]).len().sort(["round", "fas"]).rows())
print("\n== post: rounds NOT in {9,13,25,27,28} with code != within (should be none) ==")
print(post.filter(pl.col("dis") & ~pl.col("round").is_in([9, 13, 25, 27, 28])).height)
print("\n== post round 13 agreeing rows: maxr values ==")
print(post.filter((pl.col("round") == 13) & ~pl.col("dis")).group_by("maxr").len().sort("maxr").rows())
print("\n== post round 13 inverted rows: maxr values ==")
print(post.filter((pl.col("round") == 13) & pl.col("dis")).group_by("maxr").len().sort("maxr").rows())
