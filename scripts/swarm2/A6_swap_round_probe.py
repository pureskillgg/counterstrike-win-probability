"""A6 probe: which round numbers carry round_end-vs-score and round_end-vs-bomb disagreements, both eras.

For every round number r (1..30): scored rows (score-derived winner known), disagreements W != S_ns,
bomb rows, and W != B. 2024 is split by match version proxy (pure elimination codes vs has event codes)
and by the match's last round (16 = 16-round match, 24 = 24-round match, >24 = overtime match).
Reads P2 tables from the scratchpad (written by A6_threeway_probe.py).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_swap_round_probe.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
P24 = pl.read_parquet(SCR / "A6_threeway_2024.parquet")
P26 = pl.read_parquet(SCR / "A6_threeway_2026.parquet")

# match last round and version proxy (2024)
last = P24.group_by("match_key").agg(pl.col("round").max().alias("last_round"))
ev = P24.group_by("match_key").agg(pl.col("code").is_in([1, 7]).sum().alias("n_event"))
P24 = P24.join(last, on="match_key", how="left").join(ev, on="match_key", how="left").with_columns(
    pl.when(pl.col("last_round") <= 16).then(pl.lit("16r"))
    .when(pl.col("last_round") <= 24).then(pl.lit("24r"))
    .otherwise(pl.lit("OT")).alias("fmt"),
    pl.when(pl.col("n_event") == 0).then(pl.lit("elim_only")).otherwise(pl.lit("has_event")).alias("version"))
P26 = P26.join(P26.group_by("match_key").agg(pl.col("round").max().alias("last_round")), on="match_key", how="left")
P26 = P26.with_columns(
    pl.when(pl.col("last_round") <= 16).then(pl.lit("16r"))
    .when(pl.col("last_round") <= 24).then(pl.lit("24r"))
    .otherwise(pl.lit("OT")).alias("fmt"))


def table(P, label):
    print(f"== {label}")
    rows = []
    for r in range(1, 31):
        s = P.filter((pl.col("round") == r) & pl.col("S_ns").is_not_null())
        b = P.filter((pl.col("round") == r) & pl.col("B").is_not_null())
        rows.append({"round": r, "scored": s.height, "W_ne_Sns": s.filter(pl.col("W") != pl.col("S_ns")).height,
                     "bomb_rows": b.height, "W_ne_B": b.filter(pl.col("W") != pl.col("B")).height})
    out = pl.DataFrame(rows)
    print(out.filter((pl.col("scored") > 0) | (pl.col("bomb_rows") > 0)))


table(P26, "2026 all matches")
table(P24, "2024 all matches")
print("== 2024 by match format and version (round, format, version): scored, W!=Sns, bomb, W!=B")
g = P24.with_columns(
    (pl.col("W") != pl.col("S_ns")).alias("dsn"),
    (pl.col("W") != pl.col("B")).alias("dwb")).group_by(["round", "fmt", "version"]).agg(
    pl.col("S_ns").is_not_null().sum().alias("scored"),
    pl.col("dsn").sum().alias("W_ne_Sns"),
    pl.col("B").is_not_null().sum().alias("bomb_rows"),
    pl.col("dwb").sum().alias("W_ne_B")).sort(["round", "fmt", "version"])
print(g.filter((pl.col("W_ne_Sns") > 0) | (pl.col("W_ne_B") > 0)))
print("== 2024 16-round matches: round 9 by version")
print(P24.filter((pl.col("fmt") == "16r") & (pl.col("round") == 9)).group_by("version").agg(
    pl.len().alias("n"), (pl.col("W") != pl.col("S_ns")).sum().alias("W_ne_Sns")))
print("== 2024 OT matches: rounds 25..30, W != S_ns by round and version")
print(P24.filter((pl.col("fmt") == "OT") & (pl.col("round") >= 25) & (pl.col("round") <= 30)).group_by(
    ["round", "version"]).agg(pl.col("S_ns").is_not_null().sum().alias("scored"),
                              (pl.col("W") != pl.col("S_ns")).sum().alias("W_ne_Sns")).sort(["round", "version"]))
