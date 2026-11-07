"""verify D5-9 contamination: how many 2024 rounds pass the e2 'clean 5v5' filter (round_end.player_count == 20)
but belong to 2-player-per-team (roster 4) matches? Reads only scratchpad tables. Writes nothing else.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_contam.py
"""
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\VD59")
RE = pl.read_parquet(SCR / "end_raw.parquet")
ROST = pl.read_parquet(SCR / "roster.parquet")
MK = pl.read_parquet(SCR / "marks.parquet")
regA = set(MK.filter(pl.col("event_type") == "round_announce_match_start")["match_key"].to_list())

roster = {r["match_key"]: r["n_players"] for r in ROST.iter_rows(named=True)}
per_match = RE.group_by("match_key").agg(pl.len().alias("n_rounds"), (pl.col("player_count") == 20).sum().alias("n_pc20"))
per_match = per_match.with_columns(
    roster=pl.col("match_key").replace_strict(roster, default=None, return_dtype=pl.Int64),
    regime=pl.when(pl.col("match_key").is_in(list(regA))).then(pl.lit("A")).otherwise(pl.lit("B")))

print("matches with round_end rows:", per_match.height)
print("\nround_end rows with player_count == 20 (the e2 'clean 5v5' test), by roster size and regime:")
rows = per_match.filter(pl.col("n_pc20") > 0)
g_ = rows.group_by(["regime", "roster"]).agg(pl.len().alias("matches"), pl.col("n_pc20").sum().alias("rounds_pc20"))
print(g_.sort(["regime", "roster"]))
two_v_two = rows.filter((pl.col("roster") <= 5))
print("\n2v2-like matches (roster <= 5) passing the pc==20 filter:", two_v_two.height,
      " rounds:", int(two_v_two["n_pc20"].sum()), " regime:", dict(Counter(two_v_two["regime"].to_list())))
print("5v5-like matches (roster >= 8) passing the pc==20 filter:",
      rows.filter(pl.col("roster") >= 8).height, " rounds:", int(rows.filter(pl.col("roster") >= 8)["n_pc20"].sum()))
print("matches with no roster (no kills):", per_match.filter(pl.col("roster").is_null()).height)
