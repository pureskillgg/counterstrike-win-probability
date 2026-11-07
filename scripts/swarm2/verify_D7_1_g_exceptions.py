"""D7-1 verifier, part G: the 7 unflipped round-13 rows in build 9961, and the first build-9961 match."""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
S = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(S / "verify_D7_1_rounds.parquet"))
B = R.filter(pl.col("bomb_side").is_not_null() & (pl.col("player_count") == 20))
r13 = B.filter((pl.col("round") == 13) & (pl.col("build_num") == 9961))
clean = r13.filter(~pl.col("mm"))
print("G1 unflipped build-9961 round-13 bomb rows:", clean.height)
print(clean.select(["mdt", "map_name", "winner_team_code", "bomb_side", "planter_side", "win_reason_message", "tick", "out_tick", "match_key"]).with_columns(pl.col("match_key").str.slice(0, 40)))
# do these matches have other swap-round bomb rows, and do those flip?
keys = clean["match_key"].to_list()
oth = B.filter(pl.col("match_key").is_in(keys) & (pl.col("round") != 13))
print("G1 other bomb rows in these matches (round, mm):", oth.select(["match_key", "round", "mm"]).with_columns(pl.col("match_key").str.slice(0, 22)).rows())
# rounds present per match, max round for the 7
RE = R.group_by("match_key").agg(pl.col("round").max().alias("maxr"), pl.col("round").n_unique().alias("nu"))
print("G1 maxr of the 7 matches:", RE.filter(pl.col("match_key").is_in(keys)).select(["match_key", "maxr", "nu"]).with_columns(pl.col("match_key").str.slice(0, 22)).rows())
first = R.filter(pl.col("build_num") == 9961).sort("mdt").select(["mdt", "match_key"]).head(1)
fk = first["match_key"][0]
print("\nG2 first build-9961 match:", first["mdt"][0], fk[:40])
print("G2 its bomb-outcome round-13 row(s):", R.filter((pl.col("match_key") == fk) & (pl.col("round") == 13) & pl.col("bomb_side").is_not_null()).select(["round", "winner_team_code", "bomb_side", "mm", "player_count"]).rows())
print("G2 all bomb-outcome rounds of that match (round, mm, pc):", R.filter((pl.col("match_key") == fk) & pl.col("bomb_side").is_not_null()).select(["round", "mm", "player_count"]).rows())
