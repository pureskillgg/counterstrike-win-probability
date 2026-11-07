"""verify D2-5 part 5: better-powered round-label shift test (2024 and 2026).

Question: could the recorded winner R be one round off (R(r+1) carries the physical outcome of r,
or R(r-1) does)? Uses bomb outcome B(r) (in-window, from bomb_state) and the recorded R of the
neighbouring rounds, which exist for almost every round (unlike neighbouring bomb outcomes).

Defect rows: B(r) != R(r) (bomb-verified). Control rows: B(r) == R(r).
Base rate: B(r) == R(neighbour) over all bomb-verified rows of the same kind, for comparison.
A one-round shift would give B(r) == R(r+1) (or R(r-1)) near 100% in defect rows.

Imports the bomb/R table from verify_D2_5_c_recompute.py (same directory).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_5_g_shift.py
"""
import sys

import polars as pl

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D2_5_c_recompute as c  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)


def section(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


for label in ["2024", "2026"]:
    tab = c.build(label)
    # R of every full-5v5 round (no bomb needed), keyed for neighbour lookup
    rr = tab.select(["match_key", "round", pl.col("winner_team_code").alias("R_self")])
    rn = rr.with_columns((pl.col("round") - 1).alias("round")).rename({"R_self": "R_next"})   # R of r+1 at row r
    rp = rr.with_columns((pl.col("round") + 1).alias("round")).rename({"R_self": "R_prev"})   # R of r-1 at row r
    sb = tab.filter(pl.col("s_bomb").is_not_null()).select(["match_key", "round", "bin", "s_bomb", "winner_team_code"])
    j = sb.join(rn, on=["match_key", "round"], how="left").join(rp, on=["match_key", "round"], how="left")
    j = j.with_columns((pl.col("winner_team_code") != pl.col("s_bomb")).alias("defect"))
    section(f"{label}: shift test, bomb-verified rounds (n={j.height}, defect={j.filter(pl.col('defect')).height})")
    for name, g in [("defect rows", j.filter(pl.col("defect"))), ("control rows", j.filter(~pl.col("defect")))]:
        gn = g.filter(pl.col("R_next").is_not_null())
        gp = g.filter(pl.col("R_prev").is_not_null())
        print(f"{name}: n={g.height}")
        print(f"   B(r)==R(r+1): {gn.filter(pl.col('s_bomb') == pl.col('R_next')).height} of {gn.height}"
              f"   B(r)==R(r-1): {gp.filter(pl.col('s_bomb') == pl.col('R_prev')).height} of {gp.height}")
    d13 = j.filter(pl.col("defect") & (pl.col("round") == 13))
    print(f"defect rows at round 13 only: n={d13.height}"
          f"  B(13)==R(14): {d13.filter(pl.col('s_bomb') == pl.col('R_next')).height}"
          f"  B(13)==R(12): {d13.filter(pl.col('s_bomb') == pl.col('R_prev')).height}")
    # does R(r+1) equal B(r) in defect rows because R(r+1) itself is the flipped label of r+1?
    # (control: how often R(r+1) != B(r+1) at round 14, i.e. the neighbour's own defect rate)
    nb = tab.filter(pl.col("s_bomb").is_not_null()).select(["match_key", "round", "winner_team_code", "s_bomb"])
    print("neighbour-round defect rate (R != own bomb) for rounds 14 and 12 (2024 only meaningful):")
    for rr_ in [12, 14]:
        x = nb.filter(pl.col("round") == rr_)
        print(f"   round {rr_}: n={x.height} defect={x.filter(pl.col('winner_team_code') != pl.col('s_bomb')).height}")
