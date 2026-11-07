"""Null team-code deaths: are they the missing final deaths behind the 429-tick gap?

  N1. Validate attacker-side inference on coded deaths: victim side vs 5 - attacker side (cross-team kills).
  N2. Null-coded deaths: what are their attacker codes, player ids, and where do they fall in the round?
  N3. Infer victim side for null-coded deaths from the same player's side in the same round (other rows), else from attacker.
  N4. Recompute last death in the live window with null-coded deaths included (inferred side), and redo the flag test:
      flag = S != W, victim side vs S and W, death counts vs win reason.
ASSUMPTIONS:
  A10. A cross-team kill: victim side = 5 - attacker side (validated in N1 on coded rows).
  A11. Same player_id keeps one side within a round (validated by counting conflicts).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SUF = ".2023-12-28,2024-02-25"
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
import glob as _g


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    fs = sorted(_g.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


de = read("player_death", ["match_key", "round", "tick", "player_id", "attacker_id", "player_team_code",
                           "attacker_team_code", "assister_team_code", "player_id_fixed", "attacker_id_fixed"])
print("total deaths", de.height)
print("null player_team_code:", de.filter(pl.col("player_team_code").is_null()).height,
      " null player_id:", de.filter(pl.col("player_id").is_null()).height,
      " null attacker_team_code:", de.filter(pl.col("attacker_team_code").is_null()).height)

# N1: validate cross-team rule on coded rows
coded = de.filter(pl.col("player_team_code").is_in([2, 3]) & pl.col("attacker_team_code").is_in([2, 3])
                  & (pl.col("attacker_id") != pl.col("player_id")))
print("\nN1 coded kills with attacker side 2/3 and attacker != victim:", coded.height)
print("  victim side == attacker side (team kills):", coded.filter(pl.col("player_team_code") == pl.col("attacker_team_code")).height,
      " cross-team:", coded.filter(pl.col("player_team_code") != pl.col("attacker_team_code")).height)

# N2: null-coded deaths
nul = de.filter(pl.col("player_team_code").is_null())
print("\nN2 null-coded deaths: attacker_team_code value counts")
print(nul.group_by("attacker_team_code", maintain_order=False).len().sort("len", descending=True))
print("N2 null-coded deaths: player_id null?", nul.filter(pl.col("player_id").is_null()).height,
      " player_id == -1?", nul.filter(pl.col("player_id") == -1).height)
print("N2 null-coded deaths: sample rows")
print(nul.head(5))

# N3: infer victim side for null rows from the same player's side in the same round (other rows)
pairs = []
p1 = de.filter(pl.col("player_team_code").is_in([2, 3])).select(["match_key", "round", "player_id", pl.col("player_team_code").alias("side")])
p2 = de.filter(pl.col("attacker_team_code").is_in([2, 3])).select(["match_key", "round", pl.col("attacker_id").alias("player_id"), pl.col("attacker_team_code").alias("side")])
p3 = de.filter(pl.col("assister_team_code").is_in([2, 3])).select(["match_key", "round", pl.col("attacker_id").alias("player_id"), pl.col("assister_team_code").alias("side")])
lookup = pl.concat([p1, p2]).unique()
conf = lookup.group_by(["match_key", "round", "player_id"]).agg(pl.col("side").n_unique().alias("n")).filter(pl.col("n") > 1)
print("\nN3 (match,round,player) with conflicting sides in lookup:", conf.height, "of",
      lookup.group_by(["match_key", "round", "player_id"]).len().height)
lk = lookup.group_by(["match_key", "round", "player_id"]).agg(pl.col("side").first().alias("side_lk"))
nul2 = nul.join(lk, on=["match_key", "round", "player_id"], how="left")
print("null-coded deaths:", nul2.height, " resolved via same-round lookup:", nul2.filter(pl.col("side_lk").is_not_null()).height)
nul2 = nul2.with_columns(
    pl.when(pl.col("attacker_team_code").is_in([2, 3]) & (pl.col("attacker_id") != pl.col("player_id")))
    .then(5 - pl.col("attacker_team_code")).otherwise(pl.lit(None)).alias("side_att"))
print("null-coded with attacker side inferable:", nul2.filter(pl.col("side_att").is_not_null()).height)
print("agreement lookup vs 5-attacker on rows where both exist:",
      nul2.filter(pl.col("side_lk").is_not_null() & pl.col("side_att").is_not_null()).select(
          (pl.col("side_lk") == pl.col("side_att")).mean()).item())
