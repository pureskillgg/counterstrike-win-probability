"""Gap regime and null-coded deaths.

  G1. Round 13, days 12-25, full rounds with a last death in window: P(flag | gap regime), gap in {0, 429, other}.
  G2. Same by round for 9, 25, 27, 28 and controls (2, 10, 14, 19).
  G3. Null-coded deaths included: for each full round in window, add null-coded deaths with a side imputed from the
      same (match, round, player) lookup (coded rows) and from 5 - attacker side (validated in verify_D4_6_nulls.py).
      Recompute last death (coded + imputed), elimination-implied winner, and flag status. Report how many flagged rows
      change and what the new last death side is.
ASSUMPTIONS: A15. A null-coded death whose side cannot be imputed is left out of the last-death calculation (reported separately).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SUF = ".2023-12-28,2024-02-25"
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
import glob as _g


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    fs = sorted(_g.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
POP = R.filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12) & (pl.col("S") != 0)
               & (pl.col("n_victim_sides") == 1) & pl.col("victim_side").is_not_null()).with_columns(
    (pl.col("S") != pl.col("W")).alias("flag"),
    pl.when(pl.col("gap") == 0).then(pl.lit("gap0")).when(pl.col("gap") == 429).then(pl.lit("gap429")).otherwise(pl.lit("gapOther")).alias("regime"))
print("G1: round 13 (days 12-25, POP): flag rate by gap regime")
print(POP.filter(pl.col("round") == 13).group_by("regime").agg(pl.len().alias("n"), pl.col("flag").sum().alias("n_flag"),
                                                               pl.col("flag").mean().alias("P_flag")).sort("regime"))
print("\nG2: flag rate by regime, by round")
print(POP.filter(pl.col("round").is_in([9, 13, 25, 27, 28, 2, 10, 14, 19])).group_by(["round", "regime"]).agg(
    pl.len().alias("n"), pl.col("flag").mean().alias("P_flag")).sort(["round", "regime"]))

# G3: include null-coded deaths with imputed side
de = read("player_death", ["match_key", "round", "tick", "player_id", "attacker_id", "player_team_code", "attacker_team_code",
                           "assister_team_code"])
coded = de.filter(pl.col("player_team_code").is_in([2, 3]))
lk_rows = pl.concat([
    coded.select(["match_key", "round", "player_id", pl.col("player_team_code").alias("side")]),
    de.filter(pl.col("attacker_team_code").is_in([2, 3])).select(["match_key", "round", pl.col("attacker_id").alias("player_id"),
                                                                    pl.col("attacker_team_code").alias("side")]),
]).unique().group_by(["match_key", "round", "player_id"]).agg(pl.col("side").first())
nul = de.filter(pl.col("player_team_code").is_null()).join(lk_rows, on=["match_key", "round", "player_id"], how="left")
nul = nul.with_columns(
    pl.when(pl.col("attacker_team_code").is_in([2, 3]) & (pl.col("attacker_id") != pl.col("player_id")))
    .then(5 - pl.col("attacker_team_code")).otherwise(pl.lit(None)).alias("side_att"))
nul = nul.with_columns(pl.coalesce(["side", "side_att"]).alias("side_imp"))
print("\nG3 null-coded deaths:", nul.height, " imputable:", nul.filter(pl.col("side_imp").is_not_null()).height)
nul_imp = nul.filter(pl.col("side_imp").is_not_null()).select(["match_key", "round", "tick", pl.col("side_imp").alias("player_team_code")])
allw = pl.concat([coded.select(["match_key", "round", "tick", "player_team_code"]), nul_imp]).sort(["match_key", "round", "tick"])
rw = R.filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12) & (pl.col("S") != 0)).select(
    ["match_key", "round", "live", "end", "S", "W", "victim_side", "last_tick", "n_victim_sides", "gap", "win_reason_code"])
dd = allw.join(rw.select(["match_key", "round", "live", "end"]), on=["match_key", "round"], how="inner").filter(
    (pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
lt = dd.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("lt2"))
lv = dd.join(lt, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("lt2")).group_by(["match_key", "round"]).agg(
    pl.col("player_team_code").n_unique().alias("ns2"), pl.col("player_team_code").min().alias("victim2"), pl.col("lt2").first())
G = rw.join(lv, on=["match_key", "round"], how="left").filter(pl.col("ns2") == 1)
G = G.with_columns((pl.col("S") != pl.col("W")).alias("flag"), (5 - pl.col("victim2") == pl.col("S")).alias("elim_eq_S"),
                   (pl.col("victim2") != pl.col("victim_side")).alias("changed_victim"))
print("G3 rows with a last death (coded+imputed):", G.height, " changed victim vs coded-only:", G.filter(pl.col("changed_victim")).height)
print("G3 by round: flagged rows (new definition), elimination-implied winner == S share:")
print(G.filter(pl.col("round").is_in([9, 13, 25, 27, 28])).group_by(["round", "flag"]).agg(
    pl.len().alias("n"), pl.col("elim_eq_S").mean().alias("P_elim_eq_S"), pl.col("changed_victim").sum().alias("changed")).sort(["round", "flag"]))
print("G3 unflagged control rounds (2, 10, 14, 19) P(elim == S):")
print(G.filter(pl.col("round").is_in([2, 10, 14, 19])).group_by("round").agg(pl.len().alias("n"), pl.col("elim_eq_S").mean().alias("P_elim_eq_S")).sort("round"))
