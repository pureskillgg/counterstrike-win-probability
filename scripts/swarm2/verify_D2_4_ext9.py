"""Verifier D2-4: does the claim's exclusion of 9-player rounds change the coverage or the flip picture?

The claim's side rule needs 10 distinct players (5 per side). Another verifier log at this path reports
sx coverage 73,326 for 2024, above the 67,224 valid 5v5 keys in the whole tome, so it must include other rounds.
This script compares the claim's rule (valid_np=(10,)) with the extension valid_np=(9, 10), where the side rule
still needs all five team-X players on one side and all other players on the other side.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_4_ext9.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import polars as pl  # noqa: E402

from verify_D2_4_build import bin_expr, build, channel  # noqa: E402

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(16)


def hdr(s):
    print("\n" + "=" * 10 + " " + s + " " + "=" * 10)


def eq(df, a, b):
    return df.filter(pl.col(a) == pl.col(b)).height


pi = channel("2024", "player_info", ["match_key", "round", "player_id", "team_code"]).unique(
    ["match_key", "round", "player_id"], keep="first", maintain_order=True)
npc = pi.group_by(["match_key", "round"]).agg(pl.len().alias("np"))

hdr("default (claim rule, 10 players)")
b10 = build("2024").with_columns(bin_expr())
print("rounds:", b10.height, " sx:", b10.filter(pl.col("sx").is_not_null()).height,
      " score:", b10.filter(pl.col("s_score").is_not_null()).height,
      " T:", b10.filter(pl.col("T").is_not_null()).height, " flipped:", b10.filter(pl.col("flip")).height)

hdr("extension (9 or 10 players, one consistent side per team)")
b9 = build("2024", valid_np=(9, 10)).with_columns(bin_expr()).join(npc, on=["match_key", "round"], how="left")
print("rounds:", b9.height, " sx:", b9.filter(pl.col("sx").is_not_null()).height,
      " score:", b9.filter(pl.col("s_score").is_not_null()).height,
      " T:", b9.filter(pl.col("T").is_not_null()).height, " flipped:", b9.filter(pl.col("flip")).height)
ext = b9.filter(pl.col("np") == 9)
print("9-player rounds with sx:", ext.filter(pl.col("sx").is_not_null()).height,
      " with score:", ext.filter(pl.col("s_score").is_not_null()).height)

hdr("9-player extension rows: validation against bomb and elimination")
eb = ext.filter(pl.col("s_score").is_not_null() & pl.col("s_bomb").is_not_null())
print("9-player rows with score and in-window bomb:", eb.height, " score==bomb:", eq(eb, "s_score", "s_bomb"))
es = ext.filter(pl.col("s_score").is_not_null() & pl.col("s_elim").is_not_null())
print("9-player rows with score and elim:", es.height, " score==elim:", eq(es, "s_score", "s_elim"))
ef = ext.filter(pl.col("flip") & pl.col("s_score").is_not_null())
print("9-player flipped rows with score truth:", ef.height,
      " bomb known:", ef.filter(pl.col("s_bomb").is_not_null()).height,
      " bomb==T:", eq(ef.filter(pl.col("s_bomb").is_not_null()), "s_bomb", "T"))
print("9-player rounds by bin, flipped by score:")
print(ext.filter(pl.col("s_score").is_not_null()).group_by("bin").agg(
    pl.len().alias("n"), pl.col("flip").sum().alias("flipped")).sort("bin"))

hdr("default vs extension: same rounds, different truth?")
j = b10.select(["match_key", "round", pl.col("T").alias("T10"), pl.col("R")]).join(
    b9.select(["match_key", "round", pl.col("T").alias("T9")]), on=["match_key", "round"], how="inner")
both = j.filter(pl.col("T10").is_not_null() & pl.col("T9").is_not_null())
print("rounds with T in both:", both.height, " T10==T9:", eq(both, "T10", "T9"))
