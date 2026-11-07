"""Third and fourth witnesses for the flagged rounds: the bomb outcome and the scoreboard at round_officially_ended.

  B1. Bomb rounds (bomb_defused -> CT=3, bomb_exploded -> T=2; planter side checked = 2). For full 2024 rounds in
      days 12-25 with a bomb outcome: compare bomb-implied winner with S (score freeze-to-freeze) and W (round_end).
  B2. Score witness S_end: score change from round_freeze_end(r) to round_officially_ended(r) inside round_state
      (independent of the next round's freeze_end). Compare S_end with S and W, for round 13 and all rounds.
  B3. Null-coded deaths imputed (same match+round lookup, then match+half lookup): recompute the flagged set and the
      elimination-implied winner; does the flagged set shrink?
ASSUMPTIONS
  A12. bomb_exploded => T won, bomb_defused => CT won (game rule). Planter side checked in B1.
  A13. round_officially_ended is the in-game scoreboard update for round r (checked against S where S is defined).
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


R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
POP = R.filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12))

bs = read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
bo = bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).with_columns(
    pl.when(pl.col("event_type") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3)).alias("bside"))
bo = bo.group_by(["match_key", "round"]).agg(pl.col("bside").n_unique().alias("nb"), pl.col("bside").first(),
                                              pl.col("event_type").n_unique().alias("n_types"))
print("bomb outcome rows with >1 distinct outcome in a round:", bo.filter(pl.col("nb") > 1).height)
pl_side = bs.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(
    pl.col("player_team_code").first().alias("planter"))
print("planter side values (should be 2):", pl_side.group_by("planter").len().sort("planter").rows())

B = POP.join(bo.select(["match_key", "round", pl.col("bside")]), on=["match_key", "round"], how="inner")
B = B.filter(pl.col("S") != 0)
print("\nB1 bomb rounds (days 12-25, full, S!=0):", B.height)
print(B.group_by(["round"]).agg(
    pl.len().alias("n"),
    (pl.col("bside") == pl.col("S")).mean().alias("P_bomb_eq_S"),
    (pl.col("bside") == pl.col("W")).mean().alias("P_bomb_eq_W"),
    (pl.col("S") != pl.col("W")).sum().alias("n_flag"),
    ((pl.col("S") != pl.col("W")) & (pl.col("bside") == pl.col("S"))).sum().alias("flag_and_bomb_eq_S"),
    ((pl.col("S") != pl.col("W")) & (pl.col("bside") == pl.col("W"))).sum().alias("flag_and_bomb_eq_W"),
).filter(pl.col("round").is_in([9, 13, 25, 27, 28, 14, 12, 2, 3])).sort("round"))

# B2: score witness from round_officially_ended inside round_state
rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
ro = (rs.filter(pl.col("event_type") == "round_officially_ended").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").select(["match_key", "round", pl.col("t_score").alias("t_end"),
                                                            pl.col("ct_score").alias("ct_end")]))
fz = rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"]).unique(
    ["match_key", "round"], keep="first").select(["match_key", "round", pl.col("t_score").alias("t0"), pl.col("ct_score").alias("ct0")])
E = fz.join(ro, on=["match_key", "round"], how="inner").with_columns(
    pl.when((pl.col("ct_end") - pl.col("ct0") == 1) & (pl.col("t_end") - pl.col("t0") == 0)).then(pl.lit(3))
    .when((pl.col("t_end") - pl.col("t0") == 1) & (pl.col("ct_end") - pl.col("ct0") == 0)).then(pl.lit(2))
    .otherwise(pl.lit(0)).alias("S_end"))
B2 = POP.join(E.select(["match_key", "round", "S_end"]), on=["match_key", "round"], how="inner").filter(pl.col("S") != 0)
print("\nB2 score witness S_end vs S (freeze-to-freeze) and W, days 12-25 full rounds:")
print(B2.group_by("round").agg(
    pl.len().alias("n"),
    (pl.col("S_end") == pl.col("S")).mean().alias("P_Send_eq_S"),
    (pl.col("S_end") == pl.col("W")).mean().alias("P_Send_eq_W"),
    (pl.col("S_end") == 0).mean().alias("P_Send_zero"),
).filter(pl.col("round").is_in([9, 13, 25, 27, 28, 14, 12, 2, 3])).sort("round"))
print("round 13 flagged rows: S_end==S share:",
      B2.filter((pl.col("round") == 13) & (pl.col("S") != pl.col("W")))
        .select((pl.col("S_end") == pl.col("S")).mean()).item())
print("round 13 flagged rows: S_end==W share:",
      B2.filter((pl.col("round") == 13) & (pl.col("S") != pl.col("W")))
        .select((pl.col("S_end") == pl.col("W")).mean()).item())
