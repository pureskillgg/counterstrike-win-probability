"""verify D1-6 (swarm2 verifier), part H: is the bomb referee independent of team-code mapping, and are bomb events
inside the round they are attributed to?
  H1 window: bomb outcome tick inside [freeze tick, round_end tick] of the same round, mismatch vs agreeing rows (rounds 9, 13).
  H2 team codes on bomb events in the same rounds: bomb_planted player_team_code (physically T = 2) and bomb_defused
     player_team_code (physically CT = 3), in mismatch vs agreeing rows. If the mismatch rows show T planters and CT
     defusers, the team codes are physically consistent and the inverted label sits in winner_team_code alone.
  H3 bomb_exploded player_team_code (physically the planter, T = 2) for the same rows.
ASSUMPTIONS: physical side of a planter is T and of a defuser is CT (game rules); freeze tick = first round_freeze_end
tick of the round; round window = [freeze tick, round_end tick].
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(40)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def f24(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


def rd(files, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = (rd(f24("round_end"), ["match_key", "round", "tick", "winner_team_code", "player_count"])
      .unique(["match_key", "round"], keep="first", maintain_order=True).rename({"tick": "end_tick"}))
RS = rd(f24("round_state"), ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
FZ = (RS.filter(pl.col("event_type") == "round_freeze_end")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t0"),
                                            pl.col("ct_score").first().alias("c0"),
                                            pl.col("tick").min().alias("ftick")))
FZn = FZ.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("t0").alias("t1"), pl.col("c0").alias("c1")])
LED = FZ.join(FZn, on=["match_key", "round"], how="left").with_columns(
    pl.when(((pl.col("t1") - pl.col("t0")) == 1) & ((pl.col("c1") - pl.col("c0")) == 0)).then(pl.lit(2))
      .when(((pl.col("c1") - pl.col("c0")) == 1) & ((pl.col("t1") - pl.col("t0")) == 0)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("ledger_side")).select(["match_key", "round", "ledger_side", "ftick"])
A = (RE.join(LED, on=["match_key", "round"], how="left")
     .with_columns(((pl.col("ledger_side").is_not_null()) & (pl.col("ledger_side") != pl.col("winner_team_code")))
                   .alias("mm"),
                   pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
A = A.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
A = A.filter((pl.col("dg") == "d12-25") & pl.col("round").is_in([9, 13]))

BS = rd(f24("bomb_state"), ["match_key", "round", "tick", "event_type", "player_team_code"])
BS = BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
print("bomb events in d12-25 pc-agnostic rounds 9 and 13 (count by type):")
BSj = BS.join(A.select(["match_key", "round", "ftick", "end_tick", "mm", "ledger_side", "winner_team_code"]),
              on=["match_key", "round"], how="inner")
print(BSj.group_by(["event_type"]).len())
BSj = BSj.with_columns(((pl.col("tick") >= pl.col("ftick")) & (pl.col("tick") <= pl.col("end_tick"))).alias("in_window"))

hdr("H1. bomb event inside its round window (round 9 and 13, d12-25)")
print(BSj.group_by(["round", "mm", "event_type"]).agg(pl.len().alias("events"), pl.col("in_window").sum().alias("in_window"))
      .sort(["round", "mm", "event_type"]))

hdr("H2/H3. player_team_code on bomb events by mismatch status (physical expectation: planted=2 (T), defused=3 (CT), exploded=2 (T))")
print(BSj.group_by(["round", "mm", "event_type", "player_team_code"]).len().sort(["round", "mm", "event_type", "player_team_code"]))

hdr("H4. winner_team_code and ledger on the bomb-bearing mismatch rows (round 9 and 13)")
ev = (BSj.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("mm").first().alias("mm"),
                                            pl.col("ledger_side").first().alias("ledger_side"),
                                            pl.col("winner_team_code").first().alias("winner_team_code"),
                                            pl.col("in_window").all().alias("all_in_window")))
ev = ev.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
print(ev.group_by(["round", "mm"]).agg(pl.len().alias("rows_with_bomb_outcome"),
                                       (pl.col("bomb_side") == pl.col("ledger_side")).sum().alias("bomb==ledger"),
                                       (pl.col("bomb_side") == pl.col("winner_team_code")).sum().alias("bomb==winner"),
                                       pl.col("all_in_window").sum().alias("all_events_in_window"))
      .sort(["round", "mm"]))
